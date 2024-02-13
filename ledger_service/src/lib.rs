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
use cfc_crypto::{extract_key_from_cwt, PUBLIC_KEY_CLAIM};
use core::time::Duration;
use coset::{cbor::Value, cwt, cwt::ClaimsSetBuilder, CborSerializable, CoseKey, CoseSign1Builder};
use federated_compute::proto::{
    AuthorizeAccessRequest, AuthorizeAccessResponse, BlobHeader, CreateKeyRequest,
    CreateKeyResponse, DataAccessPolicy, DeleteKeyRequest, DeleteKeyResponse, Ledger,
    RevokeAccessRequest, RevokeAccessResponse,
};
use oak_attestation::dice::evidence_to_proto;
use oak_proto_rust::oak::attestation::v1::Evidence;
use oak_restricted_kernel_sdk::{EvidenceProvider, Signer};
use prost::Message;
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};

mod attestation;
mod budget;

struct PerKeyLedger {
    private_key: cfc_crypto::PrivateKey,
    public_key: Vec<u8>,
    expiration: Duration,
    budget_tracker: budget::BudgetTracker,
}

pub struct LedgerService {
    evidence: Evidence,
    signer: Box<dyn Signer>,
    current_time: Duration,
    per_key_ledgers: BTreeMap<Vec<u8>, PerKeyLedger>,
}

impl LedgerService {
    pub fn create(
        evidence_provider: Box<dyn EvidenceProvider>,
        signer: Box<dyn Signer>,
    ) -> anyhow::Result<Self> {
        // Pre-generate and convert the evidence so that we don't have to do it every time a key is
        // created.
        let evidence = evidence_to_proto(evidence_provider.get_evidence().clone())?;
        Ok(Self {
            evidence,
            signer,
            current_time: Duration::default(),
            per_key_ledgers: BTreeMap::default(),
        })
    }

    /// Updates `self.current_time` and removes expired keys.
    fn update_current_time(&mut self, now: &Option<prost_types::Timestamp>) -> anyhow::Result<()> {
        let now = Self::parse_timestamp(now).map_err(|err| anyhow!("{:?}", err))?;
        if now < self.current_time {
            return Err(anyhow!("time must be monotonic"));
        }
        self.current_time = now;
        self.per_key_ledgers.retain(|_, v| v.expiration > now);
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

    /// Parses a proto Duration as a Rust Duration.
    fn parse_duration(
        duration: &Option<prost_types::Duration>,
    ) -> Result<Duration, prost_types::DurationError> {
        duration
            .clone()
            .map_or(Ok(Duration::ZERO), <Duration>::try_from)
    }

    /// Builds a CWT containing a CoseKey.
    fn build_cwt(&self, cose_key: CoseKey, expiration: Duration) -> anyhow::Result<Vec<u8>> {
        let claims = ClaimsSetBuilder::new()
            .expiration_time(cwt::Timestamp::WholeSeconds(
                expiration.as_secs().try_into().unwrap(),
            ))
            .issued_at(cwt::Timestamp::WholeSeconds(
                self.current_time.as_secs().try_into().unwrap(),
            ))
            .private_claim(
                PUBLIC_KEY_CLAIM,
                Value::from(cose_key.to_vec().map_err(anyhow::Error::msg)?),
            )
            .build();
        CoseSign1Builder::new()
            .payload(claims.to_vec().map_err(anyhow::Error::msg)?)
            .try_create_signature(b"", |msg| Ok(self.signer.sign(msg)?.signature))?
            .build()
            .to_vec()
            .map_err(anyhow::Error::msg)
    }
}

impl Ledger for LedgerService {
    fn create_key(
        &mut self,
        request: CreateKeyRequest,
    ) -> Result<CreateKeyResponse, micro_rpc::Status> {
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
        // The expiration time cannot overflow because proto Timestamps and Durations are signed
        // but Rust's Durations are unsigned.
        let expiration = self.current_time + ttl;

        // Find an available key id. The number of keys is expected to remain small, so this is
        // unlikely to require more than 1 or 2 attempts.
        let mut key_id = vec![0u8; 4];
        while {
            OsRng.fill_bytes(key_id.as_mut_slice());
            self.per_key_ledgers.contains_key(&key_id)
        } {}

        // Construct and save a new keypair.
        let (private_key, cose_public_key) = cfc_crypto::gen_keypair(&key_id);
        let public_key = self.build_cwt(cose_public_key, expiration).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                format!("failed to encode CWT: {:?}", err),
            )
        })?;
        self.per_key_ledgers.insert(
            key_id,
            PerKeyLedger {
                private_key,
                public_key: public_key.clone(),
                expiration,
                budget_tracker: budget::BudgetTracker::new(),
            },
        );

        // Construct the response.
        Ok(CreateKeyResponse {
            public_key,
            attestation_evidence: Some(self.evidence.clone()),
            ..Default::default()
        })
    }

    fn delete_key(
        &mut self,
        request: DeleteKeyRequest,
    ) -> Result<DeleteKeyResponse, micro_rpc::Status> {
        // Extract the key id from the CoseKey inside the public key CWT.
        let key_id = extract_key_from_cwt(&request.public_key)
            .map(|key| key.key_id)
            .map_err(|err| {
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
        self.update_current_time(&request.now).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("`now` is invalid: {:?}", err),
            )
        })?;

        // Verify the attestation and compute the properties of the requesting application.
        let (recipient_app, recipient_public_key) = attestation::verify_attestation(
            &request.recipient_public_key,
            &request.recipient_attestation_evidence,
            &request.recipient_attestation_endorsements,
            &request.recipient_tag,
        )
        .map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("attestation validation failed: {:?}", err),
            )
        })?;

        // Decode the blob header and access policy. Since the access policy was provided by an
        // untrusted source, we need to verify it by checking the hash in the header. The header is
        // also unverified at this point, but will be authenticated later when it's used as the
        // associated data for re-wrapping the symmetric key. This ensures that any request that
        // uses a different header or access policy than what was approved by the client will fail.
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
        let per_key_ledger = self
            .per_key_ledgers
            .get_mut(&header.key_id)
            .ok_or_else(|| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::NotFound,
                    "public key not found",
                )
            })?;

        // Verify that the access is authorized and that there is still budget remaining.
        let transform_index = per_key_ledger.budget_tracker.find_matching_transform(
            &header.blob_id,
            header.access_policy_node_id,
            &access_policy,
            &header.access_policy_sha256,
            &recipient_app,
        )?;

        // Re-wrap the blob's symmetric key. This should be done before budgets are updated in case
        // there are decryption errors (e.g., due to invalid associated data).
        let wrap_associated_data =
            [&per_key_ledger.public_key[..], &request.recipient_nonce[..]].concat();
        let (encapsulated_key, encrypted_symmetric_key) = cfc_crypto::rewrap_symmetric_key(
            &request.encrypted_symmetric_key,
            &request.encapsulated_key,
            &per_key_ledger.private_key,
            /* unwrap_associated_data= */ &request.blob_header,
            &recipient_public_key,
            &wrap_associated_data,
        )
        .map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to re-wrap symmetric key: {:?}", err),
            )
        })?;

        // Update the budget. This shouldn't fail since there was sufficient budget earlier.
        per_key_ledger.budget_tracker.update_budget(
            &header.blob_id,
            transform_index,
            &access_policy,
            &header.access_policy_sha256,
        )?;

        // TODO(b/288282266): Include the selected transform's destination node id in the response.
        Ok(AuthorizeAccessResponse {
            encapsulated_key,
            encrypted_symmetric_key,
            reencryption_public_key: per_key_ledger.public_key.clone(),
        })
    }

    fn revoke_access(
        &mut self,
        request: RevokeAccessRequest,
    ) -> Result<RevokeAccessResponse, micro_rpc::Status> {
        let per_key_ledger = self
            .per_key_ledgers
            .get_mut(&request.key_id)
            .ok_or_else(|| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::NotFound,
                    "public key not found",
                )
            })?;

        per_key_ledger
            .budget_tracker
            .consume_budget(&request.blob_id);
        Ok(RevokeAccessResponse {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::{borrow::ToOwned, vec};
    use coset::{cwt::ClaimsSet, CoseSign1};
    use federated_compute::proto::{
        access_budget::Kind as AccessBudgetKind, data_access_policy::Transform, AccessBudget,
        ApplicationMatcher,
    };
    use oak_attestation::proto::oak::crypto::v1::Signature;
    use oak_restricted_kernel_sdk::mock_attestation::{MockEvidenceProvider, MockSigner};

    /// Macro asserting that a result is failed with a particular code and message.
    macro_rules! assert_err {
        ($left:expr, $code:expr, $substr:expr) => {
            match (&$left, &$code, &$substr) {
                (left_val, code_val, substr_val) =>
                    assert!(
                        (*left_val).as_ref().is_err_and(
                            |err| err.code == *code_val && err.message.contains(*substr_val)),
                            "assertion failed: \
                             `(val.err().code == code && val.err().message.contains(substr)`\n\
                             val: {:?}\n\
                             code: {:?}\n\
                             substr: {:?}",
                            left_val,
                            code_val,
                            substr_val)
            }
        };
    }

    /// Helper function to create a LedgerService with one key.
    fn create_ledger_service() -> (LedgerService, Vec<u8>) {
        let mut ledger = LedgerService::create(
            Box::new(MockEvidenceProvider::create().unwrap()),
            Box::new(MockSigner::create().unwrap()),
        )
        .unwrap();
        let response = ledger
            .create_key(CreateKeyRequest {
                ttl: Some(prost_types::Duration {
                    seconds: 3600,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .unwrap();
        (ledger, response.public_key)
    }

    /// Helper function to wrap a CoseKey in a CWT as would be generated by app requesting access.
    fn create_recipient_cwt(cose_key: CoseKey) -> Vec<u8> {
        let claims = ClaimsSetBuilder::new()
            .private_claim(PUBLIC_KEY_CLAIM, Value::from(cose_key.to_vec().unwrap()))
            .build();
        CoseSign1Builder::new()
            .payload(claims.to_vec().unwrap())
            .build()
            .to_vec()
            .unwrap()
    }

    #[test]
    fn test_create_key() {
        struct FakeSigner;
        impl Signer for FakeSigner {
            fn sign(&self, message: &[u8]) -> anyhow::Result<Signature> {
                return Ok(Signature {
                    signature: Sha256::digest(message).to_vec(),
                });
            }
        }
        let mut ledger = LedgerService::create(
            Box::new(MockEvidenceProvider::create().unwrap()),
            Box::new(FakeSigner),
        )
        .unwrap();

        let response1 = ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 1000,
                    ..Default::default()
                }),
                ttl: Some(prost_types::Duration {
                    seconds: 100,
                    ..Default::default()
                }),
            })
            .unwrap();
        assert!(response1.attestation_evidence.is_some());

        let cwt = CoseSign1::from_slice(&response1.public_key).unwrap();
        cwt.verify_signature(b"", |signature, message| {
            anyhow::ensure!(signature == Sha256::digest(message).as_slice());
            Ok(())
        })
        .expect("signature mismatch");
        let claims = ClaimsSet::from_slice(&cwt.payload.unwrap()).unwrap();
        assert_eq!(claims.issued_at, Some(cwt::Timestamp::WholeSeconds(1000)));
        assert_eq!(
            claims.expiration_time,
            Some(cwt::Timestamp::WholeSeconds(1100))
        );
        let key1 = extract_key_from_cwt(&response1.public_key).unwrap();

        // Since the key contains random fields, we can't check them directly. Instead, we create a
        // second key and verify that those fields are different.
        let response2 = ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 1000,
                    ..Default::default()
                }),
                ttl: Some(prost_types::Duration {
                    seconds: 100,
                    ..Default::default()
                }),
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

        // To verify that the key was actually deleted, we check that attempting to delete it again
        // produces an error.
        assert_err!(
            ledger.delete_key(DeleteKeyRequest {
                public_key,
                ..Default::default()
            }),
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
            ledger.delete_key(DeleteKeyRequest {
                public_key,
                ..Default::default()
            }),
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

        // Verify that the response contains the right public key and allows the message to be read.
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

    // TODO(b/288331695): Test authorize_access with an attestation failure.

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

        // Request access. Since `now` is after the key's expiration time, access should be denied.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 1_000_000_000,
                    ..Default::default()
                }),
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
                access_budget: Some(AccessBudget {
                    kind: Some(AccessBudgetKind::Times(1)),
                }),
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
        assert!(ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy: access_policy.clone(),
                blob_header: blob_header.clone(),
                encapsulated_key: encapsulated_key.clone(),
                encrypted_symmetric_key: encrypted_symmetric_key.clone(),
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".to_owned(),
                recipient_nonce: b"nonce1".to_vec(),
                ..Default::default()
            })
            .is_ok());

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
            micro_rpc::StatusCode::ResourceExhausted,
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
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform::default()],
            ..Default::default()
        }
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
            micro_rpc::StatusCode::ResourceExhausted,
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
    fn test_monotonic_time() {
        let (mut ledger, _) = create_ledger_service();
        ledger
            .create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 1000,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .unwrap();

        // Timestamps passed to the LedgerService must be non-decreasing.
        assert_err!(
            ledger.create_key(CreateKeyRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 500,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "time must be monotonic"
        );
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                now: Some(prost_types::Timestamp {
                    seconds: 500,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "time must be monotonic"
        );
    }
}
