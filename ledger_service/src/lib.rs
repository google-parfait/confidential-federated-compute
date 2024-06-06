// Copyright 2023 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![no_std]

extern crate alloc;

use alloc::{boxed::Box, collections::BTreeMap, format, vec, vec::Vec};
use anyhow::anyhow;
use cfc_crypto::{extract_key_from_cwt, PrivateKey, PUBLIC_KEY_CLAIM};
use core::time::Duration;
use coset::{
    cbor::Value, cwt, cwt::ClaimsSetBuilder, iana, Algorithm, CborSerializable, CoseKey,
    CoseSign1Builder, Header,
};
use federated_compute::proto::{
    AuthorizeAccessRequest, AuthorizeAccessResponse, BlobHeader, CreateKeyRequest,
    CreateKeyResponse, DataAccessPolicy, DeleteKeyRequest, DeleteKeyResponse, Ledger,
    RevokeAccessRequest, RevokeAccessResponse,
};
use hpke::{Deserializable, Serializable};
use oak_restricted_kernel_sdk::crypto::Signer;
use prost::Message;
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};

mod attestation;
mod budget;
mod test_util;

mod replication {
    include!(concat!(env!("OUT_DIR"), "/replication.rs"));
}

use crate::replication::{AuthorizeAccessEvent, CreateKeyEvent, LedgerSnapshot, PerKeySnapshot};

struct PerKeyLedger {
    private_key: PrivateKey,
    public_key: Vec<u8>,
    expiration: Duration,
    budget_tracker: budget::BudgetTracker,
}

pub struct LedgerService {
    signer: Box<dyn Signer>,
    current_time: Duration,
    per_key_ledgers: BTreeMap<Vec<u8>, PerKeyLedger>,
}

impl LedgerService {
    pub fn new(signer: Box<dyn Signer>) -> Self {
        Self { signer, current_time: Duration::default(), per_key_ledgers: BTreeMap::default() }
    }

    /// Updates `self.current_time` and removes expired keys.
    fn update_current_time(&mut self, now: &Option<prost_types::Timestamp>) -> anyhow::Result<()> {
        let now = Self::parse_timestamp(now).map_err(|err| anyhow!("{:?}", err))?;
        if now > self.current_time {
            self.current_time = now;
            self.per_key_ledgers.retain(|_, v| v.expiration > now);
        }
        Ok(())
    }

    /// Parses a proto Timestamp as a Duration since the Unix epoch.
    fn parse_timestamp(
        timestamp: &Option<prost_types::Timestamp>,
    ) -> Result<Duration, core::num::TryFromIntError> {
        timestamp.as_ref().map_or(Ok(Duration::ZERO), |ts| {
            Ok(Duration::new(ts.seconds.try_into()?, ts.nanos.try_into()?))
        })
    }

    /// Formats a Duration since the Unix epoch as a proto Timestamp.
    fn format_timestamp(timestamp: &Duration) -> Result<prost_types::Timestamp, micro_rpc::Status> {
        Ok(prost_types::Timestamp {
            seconds: timestamp.as_secs().try_into().map_err(|_| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "timestamp overflowed",
                )
            })?,
            nanos: timestamp.subsec_nanos().try_into().unwrap(),
        })
    }

    /// Parses a proto Duration as a Rust Duration.
    fn parse_duration(
        duration: &Option<prost_types::Duration>,
    ) -> Result<Duration, prost_types::DurationError> {
        duration.clone().map_or(Ok(Duration::ZERO), <Duration>::try_from)
    }

    /// Builds a CWT containing a CoseKey.
    fn build_cwt(&self, cose_key: CoseKey, expiration: Duration) -> anyhow::Result<Vec<u8>> {
        let claims = ClaimsSetBuilder::new()
            .expiration_time(cwt::Timestamp::WholeSeconds(expiration.as_secs().try_into().unwrap()))
            .issued_at(cwt::Timestamp::WholeSeconds(
                self.current_time.as_secs().try_into().unwrap(),
            ))
            .private_claim(
                PUBLIC_KEY_CLAIM,
                Value::from(cose_key.to_vec().map_err(anyhow::Error::msg)?),
            )
            .build();
        CoseSign1Builder::new()
            .protected(Header {
                alg: Some(Algorithm::Assigned(iana::Algorithm::ES256)),
                ..Default::default()
            })
            .payload(claims.to_vec().map_err(anyhow::Error::msg)?)
            .try_create_signature(b"", |msg| Ok(self.signer.sign(msg)?.signature))?
            .build()
            .to_vec()
            .map_err(anyhow::Error::msg)
    }

    /// Initiates handling of CreateKeyRequest and produces CreateKeyEvent that
    /// can be replicated and applied on all replicas of Ledger.
    pub fn produce_create_key_event(
        &mut self,
        request: CreateKeyRequest,
    ) -> Result<CreateKeyEvent, micro_rpc::Status> {
        self.update_current_time(&request.now).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("`now` is invalid: {:?}", err),
            )
        })?;

        let ttl = Self::parse_duration(&request.ttl).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("`ttl` is invalid: {:?}", err),
            )
        })?;

        // The expiration time cannot overflow because proto Timestamps and Durations
        // are signed but Rust's Durations are unsigned.
        let expiration = self.current_time + ttl;

        // Find an available key id. The number of keys is expected to remain small, so
        // this is unlikely to require more than 1 or 2 attempts.
        // This relies on the state at the time when the event is produced, so there is
        // an extremely tiny chance of key_id collision by the time when the event is
        // applied. The code that applies the event must ensure that there is no
        // collision.
        let mut key_id = vec![0u8; 4];
        while {
            OsRng.fill_bytes(key_id.as_mut_slice());
            self.per_key_ledgers.contains_key(&key_id)
        } {}

        // Construct a new keypair.
        let (private_key, cose_public_key) = cfc_crypto::gen_keypair(&key_id);
        let public_key = self.build_cwt(cose_public_key, expiration).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                format!("failed to encode CWT: {:?}", err),
            )
        })?;

        // Construct the event
        Ok(CreateKeyEvent {
            event_time: Some(Self::format_timestamp(&self.current_time)?),
            public_key,
            private_key: private_key.to_bytes().to_vec(),
            expiration: Some(Self::format_timestamp(&expiration)?),
        })
    }

    /// Applies CreateKeyEvent to the ledger state and produces
    /// CreateKeyResponse.
    pub fn apply_create_key_event(
        &mut self,
        event: CreateKeyEvent,
    ) -> Result<CreateKeyResponse, micro_rpc::Status> {
        // Update the current time.
        self.update_current_time(&event.event_time).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("event_time is invalid: {:?}", err),
            )
        })?;

        let expiration = Self::parse_timestamp(&event.expiration).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("expiration is invalid: {:?}", err),
            )
        })?;

        // Extract the key id from the CoseKey inside the public key CWT.
        let key_id =
            extract_key_from_cwt(&event.public_key).map(|key| key.key_id).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("public_key is invalid: {:?}", err),
                )
            })?;

        // Verify that there is no key_id collision
        if self.per_key_ledgers.contains_key(&key_id) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "cannot commit changes for already used key id",
            ));
        }

        let public_key = event.public_key;
        let private_key = PrivateKey::from_bytes(&event.private_key).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to parse private_key: {:?}", err),
            )
        })?;

        // Insert keys
        self.per_key_ledgers.insert(
            key_id,
            PerKeyLedger {
                private_key,
                public_key: public_key.clone(),
                expiration,
                budget_tracker: budget::BudgetTracker::new(),
            },
        );

        Ok(CreateKeyResponse { public_key, ..Default::default() })
    }

    pub fn attest_and_produce_authorize_access_event(
        &mut self,
        request: AuthorizeAccessRequest,
    ) -> Result<AuthorizeAccessEvent, micro_rpc::Status> {
        self.update_current_time(&request.now).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("`now` is invalid: {:?}", err),
            )
        })?;

        // Verify the attestation and compute the properties of the requesting
        // application.
        let (recipient_app, _) = attestation::verify_attestation(
            &request.recipient_public_key,
            request.recipient_attestation_evidence.as_ref(),
            request.recipient_attestation_endorsements.as_ref(),
            &request.recipient_tag,
        )
        .map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("attestation validation failed: {:?}", err),
            )
        })?;

        // Decode the blob header and access policy. Since the access policy was
        // provided by an untrusted source, we need to verify it by checking the
        // hash in the header. The header is also unverified at this point, but
        // will be authenticated later when it's used as the associated data for
        // re-wrapping the symmetric key. This ensures that any request that
        // uses a different header or access policy than what was approved by the client
        // will fail.
        let header = BlobHeader::decode(request.blob_header.as_ref()).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to parse blob header: {:?}", err),
            )
        })?;

        if Sha256::digest(&request.access_policy).as_slice() != header.access_policy_sha256 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "access policy does not match blob header",
            ));
        }

        let access_policy =
            DataAccessPolicy::decode(request.access_policy.as_ref()).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to parse access policy: {:?}", err),
                )
            })?;

        // Find the right per-key ledger.
        let per_key_ledger = self.per_key_ledgers.get_mut(&header.key_id).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::NotFound,
                "public key not found",
            )
        })?;

        // Verify that the access is authorized and that there is still budget
        // remaining.
        let transform_index = per_key_ledger.budget_tracker.find_matching_transform(
            &header.blob_id,
            header.access_policy_node_id,
            &access_policy,
            &header.access_policy_sha256,
            &recipient_app,
            self.current_time,
        )?;

        Ok(AuthorizeAccessEvent {
            event_time: Some(Self::format_timestamp(&self.current_time)?),
            access_policy: request.access_policy,
            transform_index: transform_index.try_into().unwrap(),
            blob_header: request.blob_header,
            encapsulated_key: request.encapsulated_key,
            encrypted_symmetric_key: request.encrypted_symmetric_key,
            recipient_public_key: request.recipient_public_key,
            recipient_nonce: request.recipient_nonce,
        })
    }

    pub fn apply_authorize_access_event(
        &mut self,
        event: AuthorizeAccessEvent,
    ) -> Result<AuthorizeAccessResponse, micro_rpc::Status> {
        // Update the current time.
        self.update_current_time(&event.event_time).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("event_time is invalid: {:?}", err),
            )
        })?;

        let recipient_public_key =
            extract_key_from_cwt(&event.recipient_public_key).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("public_key is invalid: {:?}", err),
                )
            })?;

        // Decode the blob header and the access policy.
        let header = BlobHeader::decode(event.blob_header.as_ref()).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to parse blob header: {:?}", err),
            )
        })?;

        let access_policy =
            DataAccessPolicy::decode(event.access_policy.as_ref()).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to parse access policy: {:?}", err),
                )
            })?;

        // Find the right per-key ledger.
        let per_key_ledger = self.per_key_ledgers.get_mut(&header.key_id).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::NotFound,
                "public key not found",
            )
        })?;

        // Re-wrap the blob's symmetric key. This should be done before budgets are
        // updated in case there are decryption errors (e.g., due to invalid
        // associated data).
        let wrap_associated_data =
            [&per_key_ledger.public_key[..], &event.recipient_nonce[..]].concat();
        let (encapsulated_key, encrypted_symmetric_key) = cfc_crypto::rewrap_symmetric_key(
            &event.encrypted_symmetric_key,
            &event.encapsulated_key,
            &per_key_ledger.private_key,
            /* unwrap_associated_data= */ &event.blob_header,
            &recipient_public_key,
            &wrap_associated_data,
        )
        .map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to re-wrap symmetric key: {:?}", err),
            )
        })?;

        // Update the budget. This can potentially fail if the budget is insufficient at
        // the time when the event is applied, which can be a short delay from from the
        // attestation and initially checking the budget.
        per_key_ledger.budget_tracker.update_budget(
            &header.blob_id,
            event.transform_index.try_into().unwrap(),
            &access_policy,
            &header.access_policy_sha256,
        )?;

        // TODO: b/288282266 - Include the selected transform's destination node id in
        // the response.
        Ok(AuthorizeAccessResponse {
            encapsulated_key,
            encrypted_symmetric_key,
            reencryption_public_key: per_key_ledger.public_key.clone(),
        })
    }

    /// Saves the current state into LedgerSnapshot as a part of snapshot
    /// replication.
    pub fn save_snapshot(&self) -> Result<LedgerSnapshot, micro_rpc::Status> {
        let mut snapshot = LedgerSnapshot::default();

        snapshot.current_time = Some(Self::format_timestamp(&self.current_time)?);

        for (_, per_key_ledger) in &self.per_key_ledgers {
            snapshot.per_key_snapshots.push(PerKeySnapshot {
                public_key: per_key_ledger.public_key.clone(),
                private_key: per_key_ledger.private_key.to_bytes().to_vec(),
                expiration: Some(Self::format_timestamp(&per_key_ledger.expiration)?),
                budgets: Some(per_key_ledger.budget_tracker.save_snapshot()),
            });
        }
        Ok(snapshot)
    }

    /// Replaces the current state with the state loaded from Ledger snapshot as
    /// a part of snapshot replication.
    pub fn load_snapshot(&mut self, snapshot: LedgerSnapshot) -> Result<(), micro_rpc::Status> {
        // Create a new empty map.
        let mut new_per_key_ledgers = BTreeMap::<Vec<u8>, PerKeyLedger>::default();

        for per_key_snapshot in snapshot.per_key_snapshots {
            // Create PerKeyLedger from PerKeySnapshot.
            let mut per_key_ledger = PerKeyLedger {
                private_key: PrivateKey::from_bytes(&per_key_snapshot.private_key).map_err(
                    |err| {
                        micro_rpc::Status::new_with_message(
                            micro_rpc::StatusCode::InvalidArgument,
                            format!("failed to parse private_key: {:?}", err),
                        )
                    },
                )?,
                public_key: per_key_snapshot.public_key,
                expiration: Self::parse_timestamp(&per_key_snapshot.expiration).map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("expiration is invalid: {:?}", err),
                    )
                })?,
                budget_tracker: budget::BudgetTracker::new(),
            };
            // Load the budgets.
            if per_key_snapshot.budgets.is_some() {
                per_key_ledger.budget_tracker.load_snapshot(per_key_snapshot.budgets.unwrap())?;
            }
            // Extract the key id from the CoseKey inside the public key CWT.
            let key_id = extract_key_from_cwt(&per_key_ledger.public_key)
                .map(|key| key.key_id)
                .map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("public_key is invalid: {:?}", err),
                    )
                })?;
            // Insert the PerKeyLedger with the key id.
            if new_per_key_ledgers.insert(key_id, per_key_ledger).is_some() {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "Duplicated key_id in the snapshot",
                ));
            }
        }

        // Replace the current time.
        self.current_time = Self::parse_timestamp(&snapshot.current_time).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("current_time is invalid: {:?}", err),
            )
        })?;
        // Replace the current map with the new one.
        self.per_key_ledgers = new_per_key_ledgers;
        Ok(())
    }
}

impl Ledger for LedgerService {
    fn create_key(
        &mut self,
        request: CreateKeyRequest,
    ) -> Result<CreateKeyResponse, micro_rpc::Status> {
        let create_key_event = self.produce_create_key_event(request)?;
        self.apply_create_key_event(create_key_event)
    }

    fn delete_key(
        &mut self,
        request: DeleteKeyRequest,
    ) -> Result<DeleteKeyResponse, micro_rpc::Status> {
        // Extract the key id from the CoseKey inside the public key CWT.
        let key_id =
            extract_key_from_cwt(&request.public_key).map(|key| key.key_id).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("public_key is invalid: {:?}", err),
                )
            })?;
        match self.per_key_ledgers.remove(&key_id) {
            Some(_) => Ok(DeleteKeyResponse::default()),
            None => Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::NotFound,
                "public key not found",
            )),
        }
    }

    fn authorize_access(
        &mut self,
        request: AuthorizeAccessRequest,
    ) -> Result<AuthorizeAccessResponse, micro_rpc::Status> {
        let authorize_access_event = self.attest_and_produce_authorize_access_event(request)?;
        self.apply_authorize_access_event(authorize_access_event)
    }

    fn revoke_access(
        &mut self,
        request: RevokeAccessRequest,
    ) -> Result<RevokeAccessResponse, micro_rpc::Status> {
        let per_key_ledger = self.per_key_ledgers.get_mut(&request.key_id).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::NotFound,
                "public key not found",
            )
        })?;

        per_key_ledger.budget_tracker.consume_budget(&request.blob_id);
        Ok(RevokeAccessResponse {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_err;
    use crate::attestation::{get_test_endorsements, get_test_evidence, get_test_reference_values};
    use crate::replication::{
        BlobBudgetSnapshot, BudgetSnapshot, LedgerSnapshot, PerKeySnapshot, PerPolicyBudgetSnapshot,
    };

    use alloc::{borrow::ToOwned, vec};
    use coset::{cwt::ClaimsSet, CoseSign1};
    use federated_compute::proto::{
        access_budget::Kind as AccessBudgetKind, data_access_policy::Transform, AccessBudget,
        ApplicationMatcher,
    };
    use googletest::prelude::*;
    use oak_attestation::proto::oak::crypto::v1::Signature;
    use oak_proto_rust::oak::attestation::v1::Evidence;
    use oak_restricted_kernel_sdk::testing::MockSigner;

    /// Helper function to create a LedgerService with one key.
    fn create_ledger_service() -> (LedgerService, Vec<u8>) {
        let mut ledger = LedgerService::new(Box::new(MockSigner::create().unwrap()));
        let response = ledger
            .create_key(CreateKeyRequest {
                ttl: Some(prost_types::Duration { seconds: 3600, ..Default::default() }),
                ..Default::default()
            })
            .unwrap();
        (ledger, response.public_key)
    }

    /// Helper function to wrap a CoseKey in a CWT as would be generated by app
    /// requesting access.
    fn create_recipient_cwt(cose_key: CoseKey) -> Vec<u8> {
        let claims = ClaimsSetBuilder::new()
            .private_claim(PUBLIC_KEY_CLAIM, Value::from(cose_key.to_vec().unwrap()))
            .build();
        CoseSign1Builder::new().payload(claims.to_vec().unwrap()).build().to_vec().unwrap()
    }

    #[test]
    fn test_create_key() {
        struct FakeSigner;
        impl Signer for FakeSigner {
            fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
                return Ok(Signature { signature: Sha256::digest(message).to_vec() });
            }
        }
        let mut ledger = LedgerService::new(Box::new(FakeSigner));

        let response1 = ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        let cwt = CoseSign1::from_slice(&response1.public_key).unwrap();
        cwt.verify_signature(b"", |signature, message| {
            anyhow::ensure!(signature == Sha256::digest(message).as_slice());
            Ok(())
        })
        .expect("signature mismatch");
        assert_eq!(cwt.protected.header.alg, Some(Algorithm::Assigned(iana::Algorithm::ES256)));
        let claims = ClaimsSet::from_slice(&cwt.payload.unwrap()).unwrap();
        assert_eq!(claims.issued_at, Some(cwt::Timestamp::WholeSeconds(1000)));
        assert_eq!(claims.expiration_time, Some(cwt::Timestamp::WholeSeconds(1100)));
        let key1 = extract_key_from_cwt(&response1.public_key).unwrap();

        // Since the key contains random fields, we can't check them directly. Instead,
        // we create a second key and verify that those fields are different.
        let response2 = ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();
        let key2 = extract_key_from_cwt(&response2.public_key).unwrap();
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_delete_key() {
        let (mut ledger, public_key) = create_ledger_service();
        assert_eq!(
            ledger.delete_key(DeleteKeyRequest {
                public_key: public_key.clone(),
                ..Default::default()
            }),
            Ok(DeleteKeyResponse::default())
        );

        // To verify that the key was actually deleted, we check that attempting to
        // delete it again produces an error.
        assert_err!(
            ledger.delete_key(DeleteKeyRequest { public_key, ..Default::default() }),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_delete_key_invalid() {
        let (mut ledger, _) = create_ledger_service();
        assert_err!(
            ledger.delete_key(DeleteKeyRequest {
                public_key: b"invalid".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "public_key is invalid"
        );
    }

    #[test]
    fn test_delete_key_not_found() {
        let (_, public_key) = create_ledger_service();
        let (mut ledger, _) = create_ledger_service();
        assert_err!(
            ledger.delete_key(DeleteKeyRequest { public_key, ..Default::default() }),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_authorize_access() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (ciphertext, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Request access.
        let (recipient_private_key, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let recipient_nonce: &[u8] = b"nonce";
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            })
            .unwrap();

        // Verify that the response contains the right public key and allows the message
        // to be read.
        assert_eq!(response.reencryption_public_key, public_key);
        assert_eq!(
            cfc_crypto::decrypt_message(
                &ciphertext,
                &blob_header,
                &response.encrypted_symmetric_key,
                &[&response.reencryption_public_key, recipient_nonce].concat(),
                &response.encapsulated_key,
                &recipient_private_key
            )
            .unwrap(),
            plaintext
        );
    }

    #[test]
    fn test_authorize_access_with_attestation() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (ciphertext, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Request access.
        let (recipient_private_key, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let recipient_cwt = CoseSign1Builder::new()
            .payload(
                ClaimsSetBuilder::new()
                    .private_claim(
                        PUBLIC_KEY_CLAIM,
                        Value::from(recipient_public_key.to_vec().unwrap()),
                    )
                    .build()
                    .to_vec()
                    .unwrap(),
            )
            .create_signature(b"", |message| {
                // The MockSigner signs the key with application signing key provided by the
                // MockEvidenceProvider.
                MockSigner::create().unwrap().sign(message).unwrap().signature
            })
            .build()
            .to_vec()
            .unwrap();
        let recipient_nonce: &[u8] = b"nonce";
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: recipient_cwt,
                recipient_attestation_evidence: Some(get_test_evidence()),
                recipient_attestation_endorsements: Some(get_test_endorsements()),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            })
            .unwrap();

        // Verify that the response contains the right public key and allows the message
        // to be read.
        assert_eq!(response.reencryption_public_key, public_key);
        assert_eq!(
            cfc_crypto::decrypt_message(
                &ciphertext,
                &blob_header,
                &response.encrypted_symmetric_key,
                &[&response.reencryption_public_key, recipient_nonce].concat(),
                &response.encapsulated_key,
                &recipient_private_key
            )
            .unwrap(),
            plaintext
        );
    }

    #[test]
    fn test_authorize_access_invalid_evidence() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Request access. Empty evidence will cause attestation validation to fail.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let recipient_cwt = CoseSign1Builder::new()
            .payload(
                ClaimsSetBuilder::new()
                    .private_claim(
                        PUBLIC_KEY_CLAIM,
                        Value::from(recipient_public_key.to_vec().unwrap()),
                    )
                    .build()
                    .to_vec()
                    .unwrap(),
            )
            .create_signature(b"", |message| {
                // The MockSigner signs the key with application signing key provided by the
                // MockEvidenceProvider.
                MockSigner::create().unwrap().sign(message).unwrap().signature
            })
            .build()
            .to_vec()
            .unwrap();
        let recipient_nonce: &[u8] = b"nonce";
        assert_that!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: recipient_cwt,
                recipient_attestation_evidence: Some(Evidence::default()),
                recipient_attestation_endorsements: Some(get_test_endorsements()),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            }),
            err(displays_as(contains_substring("attestation validation failed")))
        );
    }

    #[test]
    fn test_authorize_access_invalid_recipient_key() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: b"invalid".into(),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "attestation validation failed"
        );
    }

    #[test]
    fn test_authorize_access_invalid_header() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: "invalid".into(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "failed to parse blob header"
        );
    }

    #[test]
    fn test_authorize_access_invalid_access_policy_sha256() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: "invalid".into(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "access policy does not match blob header"
        );
    }

    #[test]
    fn test_authorize_access_invalid_access_policy() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that can't be decoded.
        let access_policy = b"invalid";

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy: access_policy.to_vec(),
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".into(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "failed to parse access policy"
        );
    }

    #[test]
    fn test_authorize_access_application_mismatch() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that does not grant access.
        let access_policy = DataAccessPolicy::default().encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "non-matching-tag".into(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::FailedPrecondition,
            ""
        );
    }

    #[test]
    fn test_authorize_access_decryption_error() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message that was encrypted with different associated data.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, b"other aad").unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "failed to re-wrap symmetric key"
        );
    }

    #[test]
    fn test_authorize_access_missing_key_id() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message using a public key id that doesn't exist.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.iter().chain(b"x").cloned().collect(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_authorize_access_expired_key() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access. Since `now` is after the key's expiration time, access should
        // be denied.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                now: Some(prost_types::Timestamp { seconds: 1_000_000_000, ..Default::default() }),
                access_policy,
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_authorize_access_updates_budget() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: b"blob-id".to_vec(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // The first access should succeed.
        assert!(
            ledger
                .authorize_access(AuthorizeAccessRequest {
                    access_policy: access_policy.clone(),
                    blob_header: blob_header.clone(),
                    encapsulated_key: encapsulated_key.clone(),
                    encrypted_symmetric_key: encrypted_symmetric_key.clone(),
                    recipient_public_key: create_recipient_cwt(
                        cfc_crypto::gen_keypair(b"key-id").1
                    ),
                    recipient_tag: "tag".to_owned(),
                    recipient_nonce: b"nonce1".to_vec(),
                    ..Default::default()
                })
                .is_ok()
        );

        // But the second should fail because the budget has been exhausted.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".to_owned(),
                recipient_nonce: b"nonce2".to_vec(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::PermissionDenied,
            ""
        );
    }

    #[test]
    fn test_revoke_access() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        let blob_id = b"blob-id";
        assert_eq!(
            ledger.revoke_access(RevokeAccessRequest {
                key_id: cose_key.key_id.clone(),
                blob_id: blob_id.to_vec(),
                ..Default::default()
            }),
            Ok(RevokeAccessResponse::default())
        );

        // Subsequent access should not be granted.
        let access_policy =
            DataAccessPolicy { transforms: vec![Transform::default()], ..Default::default() }
                .encode_to_vec();
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: blob_id.to_vec(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".to_owned(),
                recipient_nonce: b"nonce".to_vec(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::PermissionDenied,
            ""
        );
    }

    #[test]
    fn test_revoke_access_key_not_found() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        assert_err!(
            ledger.revoke_access(RevokeAccessRequest {
                key_id: cose_key.key_id.iter().chain(b"x").cloned().collect(),
                blob_id: "blob-id".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_produce_create_key_event_monotonic_time() {
        let mut ledger = LedgerService::new(Box::new(MockSigner::create().unwrap()));

        let event1 = ledger
            .produce_create_key_event(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        // The second request uses the `now` time that is before the `now` time from
        // the first request
        let event2 = ledger
            .produce_create_key_event(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 500, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        // Despite that the event_time should not go back and the expiration time for
        // both keys should be the same.
        assert_eq!(
            event1.event_time,
            Some(prost_types::Timestamp { seconds: 1000, ..Default::default() })
        );
        assert_eq!(
            event2.event_time,
            Some(prost_types::Timestamp { seconds: 1000, ..Default::default() })
        );
        assert_eq!(
            event1.expiration,
            Some(prost_types::Timestamp { seconds: 1100, ..Default::default() })
        );
        assert_eq!(
            event2.expiration,
            Some(prost_types::Timestamp { seconds: 1100, ..Default::default() })
        );
    }

    #[test]
    fn test_apply_create_key_event_twice() {
        let mut ledger = LedgerService::new(Box::new(MockSigner::create().unwrap()));

        let event = ledger
            .produce_create_key_event(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        // Applying the event for the first time should work.
        assert!(ledger.apply_create_key_event(event.to_owned()).is_ok());

        // Applying the same event for the second must fail due to key collision.
        assert_err!(
            ledger.apply_create_key_event(event),
            micro_rpc::StatusCode::InvalidArgument,
            "cannot commit changes for already used key id"
        );
    }

    #[test]
    fn test_apply_create_key_event_invalid_public_key() {
        let mut ledger = LedgerService::new(Box::new(MockSigner::create().unwrap()));

        let mut event = ledger
            .produce_create_key_event(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        event.public_key = b"public-key".into();
        assert_err!(
            ledger.apply_create_key_event(event),
            micro_rpc::StatusCode::InvalidArgument,
            "public_key is invalid"
        );
    }

    #[test]
    fn test_apply_create_key_event_invalid_private_key() {
        let mut ledger = LedgerService::new(Box::new(MockSigner::create().unwrap()));

        let mut event = ledger
            .produce_create_key_event(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .unwrap();

        event.private_key = b"private-key".into();
        assert_err!(
            ledger.apply_create_key_event(event),
            micro_rpc::StatusCode::InvalidArgument,
            "failed to parse private_key"
        );
    }

    #[test]
    fn test_apply_authorize_access_event_key_expired() {
        let (mut ledger, public_key) = create_ledger_service();

        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Request access - produce an event but not apply it yet.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let recipient_nonce: &[u8] = b"nonce";
        let authorize_access_event = ledger
            .attest_and_produce_authorize_access_event(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            })
            .unwrap();

        // Create another key at 4000 sec to move the time forward on the ledger.
        // This should be past expiration of the first key that was used
        // in the above authorize_access_event.
        ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp { seconds: 4000, ..Default::default() }),
                ttl: Some(prost_types::Duration { seconds: 100, ..Default::default() }),
            })
            .expect("create_key succeeded");

        // Apply authorize_access_event. It should fail because the key has now expired
        // and no longer known.
        assert_err!(
            ledger.apply_authorize_access_event(authorize_access_event),
            micro_rpc::StatusCode::NotFound,
            "public key not found"
        );
    }

    #[test]
    fn test_apply_authorize_access_event_budget_consumed() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: b"blob-id".to_vec(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Create the event for the first request without committing it.
        let authorize_access_event = ledger
            .attest_and_produce_authorize_access_event(AuthorizeAccessRequest {
                access_policy: access_policy.clone(),
                blob_header: blob_header.clone(),
                encapsulated_key: encapsulated_key.clone(),
                encrypted_symmetric_key: encrypted_symmetric_key.clone(),
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".to_owned(),
                recipient_nonce: b"nonce1".to_vec(),
                ..Default::default()
            })
            .unwrap();

        // Submit and commit the second request. This one should succeed because the
        // first request hasn't updated the budget yet.
        assert!(
            ledger
                .authorize_access(AuthorizeAccessRequest {
                    access_policy,
                    blob_header: blob_header.clone(),
                    encapsulated_key,
                    encrypted_symmetric_key,
                    recipient_public_key: create_recipient_cwt(
                        cfc_crypto::gen_keypair(b"key-id").1
                    ),
                    recipient_tag: "tag".to_owned(),
                    recipient_nonce: b"nonce2".to_vec(),
                    ..Default::default()
                })
                .is_ok()
        );

        // Now applying the event for the first request must fail.
        assert_err!(
            ledger.apply_authorize_access_event(authorize_access_event),
            micro_rpc::StatusCode::PermissionDenied,
            "no budget remaining"
        );
    }

    #[test]
    fn test_save_snapshot() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct a client message.
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();

        // Request access.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let recipient_nonce: &[u8] = b"nonce";
        let now = prost_types::Timestamp { seconds: 1000, ..Default::default() };
        let _ = ledger
            .authorize_access(AuthorizeAccessRequest {
                now: Some(now.clone()),
                access_policy: access_policy.clone(),
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            })
            .unwrap();

        // Produce the snapshot.
        let snapshot = ledger.save_snapshot().unwrap();
        assert_eq!(snapshot.per_key_snapshots.len(), 1);
        // Since the private key isn't exposed we have to assume that the one
        // in the snapshot is the right one.
        let private_key = &snapshot.per_key_snapshots[0].private_key;
        assert_eq!(
            snapshot,
            LedgerSnapshot {
                current_time: Some(now),
                per_key_snapshots: vec![PerKeySnapshot {
                    public_key,
                    private_key: private_key.clone(),
                    expiration: Some(prost_types::Timestamp {
                        seconds: 3600,
                        ..Default::default()
                    }),
                    budgets: Some(BudgetSnapshot {
                        per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                            budgets: vec![BlobBudgetSnapshot {
                                blob_id: "blob-id".into(),
                                transform_access_budgets: vec![0],
                                shared_access_budgets: vec![],
                            }]
                        }],
                        consumed_budgets: vec![],
                    }),
                }],
            }
        );
    }

    #[test]
    fn test_load_snapshot() {
        let (mut ledger, _) = create_ledger_service();
        let (private_key_1, public_key_1) = cfc_crypto::gen_keypair(b"key1");
        let (private_key_2, public_key_2) = cfc_crypto::gen_keypair(b"key2");
        let snapshot = LedgerSnapshot {
            current_time: Some(prost_types::Timestamp { seconds: 1000, ..Default::default() }),
            per_key_snapshots: vec![
                PerKeySnapshot {
                    public_key: create_recipient_cwt(public_key_1),
                    private_key: private_key_1.to_bytes().to_vec(),
                    expiration: Some(prost_types::Timestamp {
                        seconds: 2000,
                        ..Default::default()
                    }),
                    budgets: Some(BudgetSnapshot {
                        per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                            access_policy_sha256: b"hash1".to_vec(),
                            budgets: vec![BlobBudgetSnapshot {
                                blob_id: b"blob1".to_vec(),
                                ..Default::default()
                            }],
                        }],
                        consumed_budgets: vec![],
                    }),
                },
                PerKeySnapshot {
                    public_key: create_recipient_cwt(public_key_2),
                    private_key: private_key_2.to_bytes().to_vec(),
                    expiration: Some(prost_types::Timestamp {
                        seconds: 2500,
                        ..Default::default()
                    }),
                    budgets: Some(BudgetSnapshot {
                        per_policy_snapshots: vec![],
                        consumed_budgets: vec![b"blob2".to_vec()],
                    }),
                },
            ],
        };
        // Load the snapshot then save a new one and verify that the same
        // snapshot is produced.
        assert_eq!(ledger.load_snapshot(snapshot.clone()), Ok(()));
        assert_eq!(ledger.save_snapshot(), Ok(snapshot));
    }

    #[test]
    fn test_load_snapshot_replaces_state() {
        let (mut ledger, _) = create_ledger_service();
        let snapshot = LedgerSnapshot {
            current_time: Some(prost_types::Timestamp::default()),
            ..Default::default()
        };
        assert_ne!(ledger.save_snapshot(), Ok(snapshot.clone()));
        assert_eq!(ledger.load_snapshot(snapshot.clone()), Ok(()));
        assert_eq!(ledger.save_snapshot(), Ok(snapshot));
    }

    #[test]
    fn test_load_snapshot_duplicating_key_id() {
        let (mut ledger, _) = create_ledger_service();
        let (private_key, public_key) = cfc_crypto::gen_keypair(b"key-id");
        assert_err!(
            ledger.load_snapshot(LedgerSnapshot {
                current_time: Some(prost_types::Timestamp::default()),
                per_key_snapshots: vec![
                    PerKeySnapshot {
                        public_key: create_recipient_cwt(public_key.clone()),
                        private_key: private_key.to_bytes().to_vec(),
                        ..Default::default()
                    },
                    PerKeySnapshot {
                        public_key: create_recipient_cwt(public_key.clone()),
                        private_key: private_key.to_bytes().to_vec(),
                        ..Default::default()
                    }
                ],
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "Duplicated key_id in the snapshot"
        );
    }

    #[test]
    fn test_authorize_access_after_load_snapshot() {
        let (mut ledger, _) = create_ledger_service();
        let (private_key, public_key) = cfc_crypto::gen_keypair(b"key-id");
        let public_key_bytes = create_recipient_cwt(public_key.clone());

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();
        let access_policy_sha256 = Sha256::digest(&access_policy).to_vec();

        let snapshot = LedgerSnapshot {
            current_time: Some(prost_types::Timestamp { ..Default::default() }),
            per_key_snapshots: vec![PerKeySnapshot {
                public_key: public_key_bytes.clone(),
                private_key: private_key.to_bytes().to_vec(),
                expiration: Some(prost_types::Timestamp { seconds: 2000, ..Default::default() }),
                budgets: Some(BudgetSnapshot {
                    per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                        access_policy_sha256: access_policy_sha256.clone(),
                        budgets: vec![BlobBudgetSnapshot {
                            blob_id: b"blob-id".to_vec(),
                            ..Default::default()
                        }],
                    }],
                    consumed_budgets: vec![],
                }),
            }],
        };
        // Load the snapshot then save a new one and verify that the same
        // snapshot is produced.
        assert_eq!(ledger.load_snapshot(snapshot.clone()), Ok(()));

        // Now test that access can be authorized to the blob "blob-id" with
        // the same policy as above.
        let plaintext = b"plaintext";
        let blob_header = BlobHeader {
            blob_id: "blob-id".into(),
            key_id: public_key.key_id.clone(),
            access_policy_sha256,
            ..Default::default()
        }
        .encode_to_vec();
        let (ciphertext, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &public_key, &blob_header).unwrap();

        // Request access.
        let (recipient_private_key, recipient_public_key) =
            cfc_crypto::gen_keypair(b"recipient-key-id");
        let recipient_nonce: &[u8] = b"nonce";
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                recipient_nonce: recipient_nonce.to_owned(),
                ..Default::default()
            })
            .unwrap();

        // Verify that the response contains the right public key and allows the message
        // to be read.
        assert_eq!(response.reencryption_public_key, public_key_bytes);
        assert_eq!(
            cfc_crypto::decrypt_message(
                &ciphertext,
                &blob_header,
                &response.encrypted_symmetric_key,
                &[&response.reencryption_public_key, recipient_nonce].concat(),
                &response.encapsulated_key,
                &recipient_private_key
            )
            .unwrap(),
            plaintext
        );
    }
}
