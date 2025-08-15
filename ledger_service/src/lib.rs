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

#![cfg_attr(not(feature = "testing"), no_std)]
// TODO: b/392802991 - Remove this once usage of the deprecated DataAccessPolicy
// fields are removed.
#![allow(deprecated)]

extern crate alloc;

pub mod actor;
mod attestation;
mod blobid;
mod range_budget;
mod test_util;

use crate::blobid::BlobId;
use crate::range_budget::{BlobRange, BudgetTracker};

use alloc::{boxed::Box, collections::BTreeMap, format, vec, vec::Vec};
use anyhow::anyhow;
use cfc_crypto::{extract_key_from_cwt, PrivateKey, PUBLIC_KEY_CLAIM};
use core::time::Duration;
use coset::{
    cbor::Value, cwt, cwt::ClaimsSetBuilder, iana, Algorithm, CborSerializable, CoseKey,
    CoseSign1Builder, Header,
};
use federated_compute::proto::{
    authorize_access_response::AuthorizedBlobKeys, pipeline_variant_policy, AuthorizeAccessRequest,
    AuthorizeAccessResponse, BlobHeader, CreateKeyRequest, CreateKeyResponse, DataAccessPolicy,
    DeleteKeyRequest, DeleteKeyResponse, Ledger, LogicalPipelinePolicy, PipelineVariantPolicy,
    RevokeAccessRequest, RevokeAccessResponse, Status,
};
use hashbrown::HashSet;
use hpke::{Deserializable, Serializable};
use oak_crypto::signer::Signer;
use prost::Message;
use rand::{rngs::OsRng, RngCore};
use rangemap::StepLite;
use sha2::{Digest, Sha256};

mod replication {
    include!(concat!(env!("OUT_DIR"), "/replication.rs"));
}

#[cfg(feature = "testing")]
pub use attestation::{
    get_test_endorsements, get_test_evidence, get_test_reference_values, get_test_signer,
};

use crate::replication::{
    authorize_access_event::BlobMetadata, AuthorizeAccessEvent, CreateKeyEvent, LedgerSnapshot,
    PerKeySnapshot, Range,
};

struct PerKeyLedger {
    private_key: PrivateKey,
    public_key: Vec<u8>,
    expiration: Duration,
    budget_tracker: BudgetTracker,
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
            .try_create_signature(b"", |msg| Ok::<Vec<u8>, anyhow::Error>(self.signer.sign(msg)))?
            .build()
            .to_vec()
            .map_err(anyhow::Error::msg)
    }

    /// Converts a legacy DataAccessPolicy with root-level Transforms to a
    /// modern DataAccessPolicy with a single LogicalPipelinePolicy containing a
    /// single PipelineVariantPolicy.
    fn convert_legacy_policy_to_modern(legacy_policy: DataAccessPolicy) -> DataAccessPolicy {
        // Create a new PipelineVariantPolicy from the legacy fields.
        let new_pipeline_variant = PipelineVariantPolicy {
            transforms: legacy_policy
                .transforms
                .into_iter()
                .map(|legacy_transform| {
                    // Convert legacy DataAccessPolicy::Transform to
                    // PipelineVariantPolicy::Transform.
                    pipeline_variant_policy::Transform {
                        src_node_ids: vec![legacy_transform.src],
                        application: legacy_transform.application,
                        access_budget: legacy_transform.access_budget,
                        shared_access_budget_indices: legacy_transform.shared_access_budget_indices,
                        ..Default::default()
                    }
                })
                .collect(),
            shared_access_budgets: legacy_policy.shared_access_budgets,
        };

        // Create a new LogicalPipelinePolicy with the new PipelineVariantPolicy.
        let new_logical_pipeline = LogicalPipelinePolicy { instances: vec![new_pipeline_variant] };

        // Create a new DataAccessPolicy with the new LogicalPipelinePolicy.
        DataAccessPolicy {
            pipelines: BTreeMap::from([("LOGICAL_PIPELINE".into(), new_logical_pipeline)]),
            ..Default::default()
        }
    }

    /// Given a modern DataAccessPolicy, extract the PipelineVariantPolicy.
    /// The ledger currently only supports DataAccessPolicies with a single
    /// LogicalPipelinePolicy with a single PipelineVariantPolicy.
    fn extract_pipeline_variant(
        policy: &DataAccessPolicy,
    ) -> Result<&PipelineVariantPolicy, micro_rpc::Status> {
        // Ensure there's exactly one LogicalPipelinePolicy.
        if policy.pipelines.len() != 1 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "DataAccessPolicy must have a single LogicalPipelinePolicy.",
            ));
        }

        // Get the first LogicalPipelinePolicy in the map.
        let logical_pipeline = policy.pipelines.values().next().ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                "Failed to get the first LogicalPipelinePolicy.",
            )
        })?;

        // Ensure there's exactly one PipelineVariantPolicy.
        if logical_pipeline.instances.len() != 1 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "The LogicalPipelinePolicy must have a single PipelineVariantPolicy.",
            ));
        }

        // Get the first (and only) PipelineVariantPolicy.
        // Clone the PipelineVariantPolicy to return an owned value.
        logical_pipeline.instances.first().ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                "Failed to get the first PipelineVariantPolicy.",
            )
        })
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
                budget_tracker: BudgetTracker::default(),
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

        // Decode the access policy.
        let mut decoded_access_policy = DataAccessPolicy::decode(request.access_policy.as_ref())
            .map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to parse access policy: {:?}", err),
                )
            })?;

        // If it's still a legacy policy, convert it to a modern policy. From this point
        // onward, we'll only consider the decoded_access_policy.pipelines field, and
        // ignore all of the deprecated DataAccessPolicy fields.
        if decoded_access_policy.pipelines.is_empty() {
            decoded_access_policy = Self::convert_legacy_policy_to_modern(decoded_access_policy);
        }

        let access_policy = decoded_access_policy;

        // Find the right `access_policy_node_id`.
        // TODO: Modify AuthorizeAccessRequest to include the `access_policy_node_id` as
        // a top level field to avoid parsing blob headers below.
        let mut access_policy_node_id = None;
        let mut blob_metadata: Vec<BlobMetadata> = Vec::with_capacity(request.blob_metadata.len());
        let legacy_mode = request.blob_metadata.len() == 0;
        if legacy_mode {
            let header = BlobHeader::decode(request.blob_header.as_ref()).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to parse blob header: {:?}", err),
                )
            })?;
            access_policy_node_id = Some(header.access_policy_node_id);
        } else {
            for blob in request.blob_metadata {
                let node_ids_same = BlobHeader::decode(blob.blob_header.as_ref()).map_or_else(
                    // Ignore decoding error since it will be dealt with later when applying
                    // the event.
                    |_err| true,
                    |header| {
                        if access_policy_node_id.is_none() {
                            access_policy_node_id = Some(header.access_policy_node_id);
                        }
                        access_policy_node_id == Some(header.access_policy_node_id)
                    },
                );

                if !node_ids_same {
                    return Err(micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("access_policy_node_id is not same for all blobs in the range"),
                    ));
                }

                blob_metadata.push(BlobMetadata {
                    blob_header: blob.blob_header,
                    encapsulated_key: blob.encapsulated_key,
                    encrypted_symmetric_key: blob.encrypted_symmetric_key,
                    recipient_nonce: blob.recipient_nonce,
                });
            }
        }

        if access_policy_node_id.is_none() {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "no access_policy_node_id found.",
            ));
        }

        let pipeline_variant_policy = Self::extract_pipeline_variant(&access_policy)?;

        // Verify that the access is authorized and get the first matching transform.
        let transform_index = BudgetTracker::find_matching_transform(
            access_policy_node_id.unwrap(),
            &pipeline_variant_policy,
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
            blob_range: request.blob_range.map(|r| Range { start: r.start, end: r.end }),
            blob_metadata,
            ..Default::default()
        })
    }

    pub fn apply_authorize_access_event(
        &mut self,
        event: AuthorizeAccessEvent,
        rewrap_keys: bool,
    ) -> Result<AuthorizeAccessResponse, micro_rpc::Status> {
        // Update the current time.
        self.update_current_time(&event.event_time).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("event_time is invalid: {:?}", err),
            )
        })?;

        // Extract the recipient public key from the event.
        let recipient_public_key =
            extract_key_from_cwt(&event.recipient_public_key).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("public_key is invalid: {:?}", err),
                )
            })?;

        // Build the vector of blobs in the batch.
        // The legacy mode is true when a single blob is specified at the top of the
        // event as opposed to the repeated blob_metadata field.
        let (blob_metadata, legacy_mode) = if event.blob_metadata.len() > 0 {
            (event.blob_metadata, false)
        } else {
            (
                vec![BlobMetadata {
                    blob_header: event.blob_header,
                    encapsulated_key: event.encapsulated_key,
                    encrypted_symmetric_key: event.encrypted_symmetric_key,
                    recipient_nonce: event.recipient_nonce,
                }],
                true,
            )
        };

        // Compute policy hash.
        let access_policy_sha256 = Sha256::digest(&event.access_policy).to_vec();

        // Decode the access policy.
        let mut decoded_access_policy = DataAccessPolicy::decode(event.access_policy.as_ref())
            .map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to parse access policy: {:?}", err),
                )
            })?;

        // If it's still a legacy policy, convert it to a modern policy. From this point
        // onward, we'll only consider the decoded_access_policy.pipelines field, and
        // ignore all of the deprecated DataAccessPolicy fields.
        if decoded_access_policy.pipelines.is_empty() {
            decoded_access_policy = Self::convert_legacy_policy_to_modern(decoded_access_policy);
        }
        let pipeline_variant_policy = Self::extract_pipeline_variant(&decoded_access_policy)?;

        // Parse the blob range or derive it from blob_id if there is exactly one blob.
        let blob_range = match event.blob_range {
            Some(range) => {
                let start = BlobId::from_vec(&range.start).map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("Invalid range start `blob_id` err:{:?}", err),
                    )
                })?;
                let end = BlobId::from_vec(&range.end).map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("Invalid range end `blob_id` err:{:?}", err),
                    )
                })?;
                BlobRange { start, end }
            }
            None => {
                if blob_metadata.len() > 1 {
                    return Err(micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("the range is required unless there is exactly one input blob"),
                    ));
                }
                let header =
                    BlobHeader::decode(blob_metadata[0].blob_header.as_ref()).map_err(|err| {
                        micro_rpc::Status::new_with_message(
                            micro_rpc::StatusCode::InvalidArgument,
                            format!("failed to parse blob header: {:?}", err),
                        )
                    })?;
                let blob_id = BlobId::from_vec(&header.blob_id).map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("Invalid `blob_id`: {:?}", err),
                    )
                })?;
                if blob_id == BlobId::MAX {
                    return Err(micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("blob_id must be strictly between [BlobId::MIN, BlobId::MAX)."),
                    ));
                }
                BlobRange { start: blob_id.clone(), end: blob_id.add_one() }
            }
        };

        // This vector holds per-blob responses with the same ordering as the original
        // blob_metadata field. For legacy mode, this vector contains exactly one entry.
        let mut authorized_blob_keys: Vec<AuthorizedBlobKeys> = Vec::default();
        authorized_blob_keys.resize_with(blob_metadata.len(), Default::default);
        // This map is used to sort indices of authorized blobs per key_id.
        let mut blob_indices_per_key_id = BTreeMap::<Vec<u8>, Vec<usize>>::default();
        // IDs of visited blobs to ensure that there are no duplicating blobs.
        let mut blob_ids = HashSet::new();
        // The total number of blobs that have been authorized either successfully or
        // not.
        let mut num_authorized_blobs = 0;
        // Authorize all blobs.
        for i in 0..blob_metadata.len() {
            match self.authorize_blob_access(
                &blob_metadata[i],
                &pipeline_variant_policy,
                &access_policy_sha256,
                event.transform_index.try_into().unwrap(),
                &mut blob_ids,
                &blob_range,
            ) {
                Ok(ledger_key_id) => {
                    blob_indices_per_key_id.entry(ledger_key_id).or_default().push(i);
                }
                Err(error) => {
                    if legacy_mode {
                        return Err(error);
                    }
                    authorized_blob_keys[i].status =
                        Some(Status { code: error.code as i32, message: error.message.into() });
                    num_authorized_blobs += 1;
                }
            };
        }

        // Find the earliest expiring ledger key to be used for reencrypting derived
        // objects.
        let mut reencryption_public_key = Vec::default();
        if rewrap_keys {
            let mut reencryption_public_key_expiration: Duration = Duration::MAX;
            for (key_id, _) in &blob_indices_per_key_id {
                let per_key_ledger = self.get_per_key_ledger(key_id)?;
                if per_key_ledger.expiration < reencryption_public_key_expiration {
                    reencryption_public_key_expiration = per_key_ledger.expiration;
                    reencryption_public_key = per_key_ledger.public_key.clone();
                }
            }
        }

        for (key_id, blob_indices) in blob_indices_per_key_id {
            let per_key_ledger = self.get_per_key_ledger(&key_id)?;
            let private_key = &per_key_ledger.private_key;
            // Rewrap symmetric keys for every blob for the current ledger key.
            if rewrap_keys {
                for i in blob_indices {
                    match Self::rewrap_symmetric_key(
                        &blob_metadata[i],
                        &reencryption_public_key,
                        private_key,
                        &recipient_public_key,
                    ) {
                        Ok((encapsulated_key, encrypted_symmetric_key)) => {
                            authorized_blob_keys[i] = AuthorizedBlobKeys {
                                encapsulated_key,
                                encrypted_symmetric_key,
                                status: Some(Status { code: 0, ..Default::default() }),
                                ..Default::default()
                            };
                            num_authorized_blobs += 1;
                        }
                        Err(error) => {
                            if legacy_mode {
                                return Err(error);
                            }
                            authorized_blob_keys[i].status = Some(Status {
                                code: error.code as i32,
                                message: error.message.into(),
                            });
                            num_authorized_blobs += 1;
                        }
                    };
                }
            }
            let mut policy_budget_tracker = per_key_ledger.budget_tracker.get_policy_budget(
                &access_policy_sha256,
                &pipeline_variant_policy,
                event.transform_index.try_into().unwrap(),
            )?;
            // Per key, update the budget once for the entire range.
            policy_budget_tracker.update_budget(&blob_range);
        }

        if !rewrap_keys {
            return Ok(AuthorizeAccessResponse::default());
        }

        if num_authorized_blobs != blob_metadata.len() {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "num_authorized_blobs must match blob_metadata.len(). This is likely an internal bug.",
            ));
        }

        // TODO: b/288282266 - Include the selected transform's destination node id in
        // the response.
        if legacy_mode {
            let first_authorized_blob = authorized_blob_keys.pop().unwrap();
            Ok(AuthorizeAccessResponse {
                encapsulated_key: first_authorized_blob.encapsulated_key,
                encrypted_symmetric_key: first_authorized_blob.encrypted_symmetric_key,
                reencryption_public_key,
                ..Default::default()
            })
        } else {
            Ok(AuthorizeAccessResponse {
                authorized_blob_keys,
                reencryption_public_key,
                ..Default::default()
            })
        }
    }

    fn get_per_key_ledger(
        &mut self,
        key_id: &Vec<u8>,
    ) -> Result<&mut PerKeyLedger, micro_rpc::Status> {
        self.per_key_ledgers.get_mut(key_id).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::NotFound,
                "public key not found",
            )
        })
    }

    fn authorize_blob_access(
        &mut self,
        blob: &BlobMetadata,
        access_policy: &PipelineVariantPolicy,
        access_policy_sha256: &Vec<u8>,
        transform_index: usize,
        blob_ids: &mut HashSet<BlobId>,
        range: &BlobRange,
    ) -> Result<Vec<u8>, micro_rpc::Status> {
        // Decode the blob header.
        let header = BlobHeader::decode(blob.blob_header.as_ref()).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to parse blob header: {:?}", err),
            )
        })?;

        // Find the right per-key ledger.
        let per_key_ledger = self.get_per_key_ledger(&header.key_id)?;

        // Verify that all blobs use the same policy.
        if header.access_policy_sha256 != *access_policy_sha256 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "access policy does not match blob header",
            ));
        }

        // Parse the blob ID.
        let blob_id = BlobId::from_vec(&header.blob_id).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("Invalid `blob_id`: {:?}", err),
            )
        })?;

        // Verify that the blob_id is unique.
        if !blob_ids.insert(blob_id) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("Duplicate blob_id {:?} found", blob_id),
            ));
        }

        // Verify that the blob_id is within the specified range.
        // Note that end is exclusive so blob_id must be strictly less that `range.end`.
        if blob_id < range.start || blob_id >= range.end {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("{:?} found outside range [{:?}, {:?})", blob_id, range.start, range.end),
            ));
        }

        // Get the policy budget tracker.
        let policy_budget_tracker = per_key_ledger.budget_tracker.get_policy_budget(
            &access_policy_sha256,
            &access_policy,
            transform_index,
        )?;

        // Check if there is remaining budget.
        if !policy_budget_tracker.has_budget(&blob_id) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "no budget remaining",
            ));
        }

        Ok(header.key_id.clone())
    }

    fn rewrap_symmetric_key(
        blob: &BlobMetadata,
        public_key: &Vec<u8>,
        private_key: &PrivateKey,
        recipient_public_key: &CoseKey,
    ) -> Result<(Vec<u8>, Vec<u8>), micro_rpc::Status> {
        // Re-wrap the blob's symmetric key. This should be done before budgets are
        // updated in case there are decryption errors (e.g., due to invalid
        // associated data).
        let wrap_associated_data = [&public_key[..], &blob.recipient_nonce[..]].concat();
        cfc_crypto::rewrap_symmetric_key(
            &blob.encrypted_symmetric_key,
            &blob.encapsulated_key,
            &private_key,
            /* unwrap_associated_data= */ &blob.blob_header,
            recipient_public_key,
            &wrap_associated_data,
        )
        .map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to re-wrap symmetric key: {:?}", err),
            )
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
                budget_tracker: BudgetTracker::default(),
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
        self.apply_authorize_access_event(authorize_access_event, true)
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

        let blob_id = BlobId::from_vec(&request.blob_id).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("Invalid `blob_id`: {:?}", err),
            )
        })?;

        per_key_ledger.budget_tracker.revoke(&blob_id);
        Ok(RevokeAccessResponse {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_err;
    use crate::attestation::{get_test_endorsements, get_test_evidence, get_test_reference_values};
    use crate::replication::{
        BudgetSnapshot, LedgerSnapshot, PerKeySnapshot, PerPolicyBudgetSnapshot,
        RangeBudgetSnapshot,
    };

    use alloc::{borrow::ToOwned, vec};
    use coset::{cwt::ClaimsSet, CoseSign1};
    use federated_compute::proto::{
        access_budget::Kind as AccessBudgetKind, authorize_access_request::BlobMetadata,
        authorize_access_request::Range, data_access_policy::Transform, AccessBudget,
        ApplicationMatcher,
    };
    use googletest::prelude::*;
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
            fn sign(&self, message: &[u8]) -> Vec<u8> {
                Sha256::digest(message).to_vec()
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
    fn test_authorize_access_multiple_blobs() {
        // Create 3 public keys.
        let mut ledger_public_key = Vec::with_capacity(3);
        let (mut ledger, public_key) = create_ledger_service();
        ledger_public_key.push(public_key);
        ledger_public_key.push(
            ledger
                .create_key(CreateKeyRequest {
                    ttl: Some(prost_types::Duration { seconds: 7200, ..Default::default() }),
                    ..Default::default()
                })
                .unwrap()
                .public_key,
        );
        ledger_public_key.push(
            ledger
                .create_key(CreateKeyRequest {
                    ttl: Some(prost_types::Duration { seconds: 10800, ..Default::default() }),
                    ..Default::default()
                })
                .unwrap()
                .public_key,
        );
        let mut cose_key = Vec::with_capacity(3);
        cose_key.push(extract_key_from_cwt(&ledger_public_key[0]).unwrap());
        cose_key.push(extract_key_from_cwt(&ledger_public_key[1]).unwrap());
        cose_key.push(extract_key_from_cwt(&ledger_public_key[2]).unwrap());

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

        // Construct 6 client messages, 2 per key.
        let plaintext = b"plaintext";
        let mut blob_header = Vec::with_capacity(6);
        let mut ciphertexts = Vec::with_capacity(6);
        let mut blob_metadata = Vec::with_capacity(6);
        let recipient_nonce: &[u8] = b"nonce";
        let mut key_index = 0;
        let mut curr_count = 0;
        for i in 0..6 {
            blob_header.push(
                BlobHeader {
                    blob_id: BlobId::from(i as u128).to_vec(),
                    key_id: cose_key[key_index].key_id.clone(),
                    access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                    ..Default::default()
                }
                .encode_to_vec(),
            );

            let (ciphertext, encapsulated_key, encrypted_symmetric_key) =
                cfc_crypto::encrypt_message(plaintext, &cose_key[key_index], &blob_header[i])
                    .unwrap();

            blob_metadata.push(BlobMetadata {
                blob_header: blob_header[i].clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_nonce: recipient_nonce.to_vec(),
            });
            ciphertexts.push(ciphertext);

            curr_count += 1;
            if curr_count == 2 {
                key_index += 1;
                curr_count = 0;
            }
        }
        // Add some blobs with invalid header and invalid policy hash.
        blob_metadata.push(BlobMetadata {
            blob_header: "invalid".into(),
            encapsulated_key: "encapsulated_key".into(),
            encrypted_symmetric_key: "encrypted_symmetric_key".into(),
            recipient_nonce: recipient_nonce.to_vec(),
        });
        blob_metadata.push(BlobMetadata {
            blob_header: BlobHeader {
                blob_id: BlobId::from(4).to_vec(),
                key_id: cose_key[1].key_id.clone(),
                access_policy_sha256: "invalid".into(),
                ..Default::default()
            }
            .encode_to_vec(),
            encapsulated_key: "encapsulated_key".into(),
            encrypted_symmetric_key: "encrypted_symmetric_key".into(),
            recipient_nonce: recipient_nonce.to_vec(),
        });

        // Request access.
        let (recipient_private_key, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                // Since `now` is after the first key's expiration time, access should be denied for
                // the first 2 blobs.
                now: Some(prost_types::Timestamp { seconds: 4000, ..Default::default() }),
                access_policy,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata,
                blob_range: Some(Range {
                    start: BlobId::from(0).to_vec(),
                    end: BlobId::from(8).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();

        // The reencryption key must be the earliest expiring key from the batch of
        // authorized blobs.
        let expected_reencryption_key = ledger_public_key[1].clone();
        // Verify that the response contains the right public key and allows the message
        // to be read.
        assert_eq!(response.reencryption_public_key, expected_reencryption_key);
        assert_eq!(response.authorized_blob_keys.len(), 8);
        for i in 2..6 {
            let authorized_blob_key = response.authorized_blob_keys.get(i).unwrap();
            assert_eq!(
                authorized_blob_key.status.as_ref().unwrap().code,
                micro_rpc::StatusCode::Ok as i32
            );
            assert_eq!(
                cfc_crypto::decrypt_message(
                    &ciphertexts[i],
                    &blob_header[i],
                    &authorized_blob_key.encrypted_symmetric_key,
                    &[&expected_reencryption_key, recipient_nonce].concat(),
                    &authorized_blob_key.encapsulated_key,
                    &recipient_private_key
                )
                .unwrap(),
                plaintext
            );
        }
        // First 2 blobs must correspond to the expired key.
        for i in 0..2 {
            assert_eq!(
                response.authorized_blob_keys.get(i).unwrap().status.as_ref().unwrap().code,
                micro_rpc::StatusCode::NotFound as i32
            );
        }
        // Last 2 responses must correspond to invalid blobs.
        for i in 6..8 {
            assert_eq!(
                response.authorized_blob_keys.get(i).unwrap().status.as_ref().unwrap().code,
                micro_rpc::StatusCode::InvalidArgument as i32
            );
        }
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
            .create_signature(b"", |message| get_test_signer().sign(message))
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

    // Uses the new schema for DataAccessPolicies
    #[test]
    fn test_authorize_access_authorized_logical_pipelines() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        // Define an access policy that grants access.
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            pipelines: {
                BTreeMap::from([(
                    "LOGICAL_PIPELINE".into(),
                    LogicalPipelinePolicy {
                        instances: vec![PipelineVariantPolicy {
                            transforms: vec![pipeline_variant_policy::Transform {
                                src_node_ids: vec![0],
                                application: Some(ApplicationMatcher {
                                    tag: Some(recipient_tag.to_owned()),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                )])
            },
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
            .create_signature(b"", |message| get_test_signer().sign(message))
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
    fn test_authorize_access_invalid_header_single_blob() {
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
    fn test_authorize_access_no_range_multiple_blobs_invalid() {
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

        // Construct multiple client messages
        let plaintext = b"plaintext";
        let mut blob_metadata = Vec::with_capacity(4);
        let recipient_nonce: &[u8] = b"nonce";
        for i in 0..4 {
            let blob_header = BlobHeader {
                blob_id: BlobId::from(i as u128).to_vec(),
                key_id: cose_key.key_id.clone(),
                access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                ..Default::default()
            }
            .encode_to_vec();

            let (_, encapsulated_key, encrypted_symmetric_key) =
                cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

            blob_metadata.push(BlobMetadata {
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_nonce: recipient_nonce.to_vec(),
            });
        }

        // Request access.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata,
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "the range is required unless there is exactly one input blob"
        );
    }

    #[test]
    fn test_authorize_access_duplicate_blob_ids() {
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

        // Construct blob_metadata.
        let blob_header = BlobHeader {
            blob_id: BlobId::from(0).to_vec(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header).unwrap();
        let blob_metadata = BlobMetadata {
            blob_header: blob_header.clone(),
            encapsulated_key,
            encrypted_symmetric_key,
            recipient_nonce: b"nonce".to_vec(),
        };

        // Request access.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata: vec![blob_metadata.clone(), blob_metadata.clone()],
                blob_range: Some(Range {
                    start: BlobId::from(0).to_vec(),
                    end: BlobId::from(1).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(response.reencryption_public_key, public_key);
        assert_eq!(response.authorized_blob_keys.len(), 2);
        // First blob should be successful.
        let authorized_blob_key_1 = response.authorized_blob_keys.get(0).unwrap();
        assert!(!authorized_blob_key_1.encapsulated_key.is_empty());
        assert!(!authorized_blob_key_1.encrypted_symmetric_key.is_empty());
        assert_eq!(
            authorized_blob_key_1.status.as_ref().unwrap().code,
            micro_rpc::StatusCode::Ok as i32
        );

        // Second should fail because it was repeated.
        let authorized_blob_key_2 = response.authorized_blob_keys.get(1).unwrap();
        assert!(authorized_blob_key_2.encapsulated_key.is_empty());
        assert!(authorized_blob_key_2.encrypted_symmetric_key.is_empty());
        assert_eq!(
            authorized_blob_key_2.status.as_ref().unwrap().code,
            micro_rpc::StatusCode::InvalidArgument as i32
        );
        assert_eq!(
            authorized_blob_key_2.status.as_ref().unwrap().message,
            "Duplicate blob_id BlobId { id: 0 } found",
        );
    }

    #[test]
    fn test_authorize_access_blob_id_outside_range() {
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

        // Construct 5 client messages with blob_ids 0, 1, 2, 3, 4.
        let plaintext = b"plaintext";
        let mut blob_metadata = Vec::with_capacity(5);
        let recipient_nonce: &[u8] = b"nonce";
        for i in 0..5 {
            let blob_header = BlobHeader {
                blob_id: BlobId::from(i as u128).to_vec(),
                key_id: cose_key.key_id.clone(),
                access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                ..Default::default()
            }
            .encode_to_vec();

            let (_, encapsulated_key, encrypted_symmetric_key) =
                cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

            blob_metadata.push(BlobMetadata {
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_nonce: recipient_nonce.to_vec(),
            });
        }

        // Request access for range 1-4 so blobs 0 and 4 should lie outside the range.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");
        let response = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata,
                blob_range: Some(Range {
                    start: BlobId::from(1).to_vec(),
                    end: BlobId::from(4).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(response.reencryption_public_key, public_key);
        assert_eq!(response.authorized_blob_keys.len(), 5);
        for i in 1..4 {
            let authorized_blob_key = response.authorized_blob_keys.get(i).unwrap();
            assert_eq!(
                authorized_blob_key.status.as_ref().unwrap().code,
                micro_rpc::StatusCode::Ok as i32
            );
            assert!(!authorized_blob_key.encapsulated_key.is_empty());
            assert!(!authorized_blob_key.encrypted_symmetric_key.is_empty());
        }
        for i in [0, 4] {
            let authorized_blob_key = response.authorized_blob_keys.get(i).unwrap();
            assert_eq!(
                authorized_blob_key.status.as_ref().unwrap().code,
                micro_rpc::StatusCode::InvalidArgument as i32
            );
            assert!(authorized_blob_key
                .status
                .as_ref()
                .unwrap()
                .message
                .ends_with("found outside range [BlobId { id: 1 }, BlobId { id: 4 })"));
            assert!(authorized_blob_key.encapsulated_key.is_empty());
            assert!(authorized_blob_key.encrypted_symmetric_key.is_empty());
        }
    }

    #[test]
    fn test_authorize_access_all_invalid_headers_multiple_blobs() {
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

        // Construct client messages.
        let blob_header_1 = BlobHeader {
            blob_id: "blob-id-1".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key_1, encrypted_symmetric_key_1) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header_1).unwrap();
        let blob_header_2 = BlobHeader {
            blob_id: "blob-id-2".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key_2, encrypted_symmetric_key_2) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header_2).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata: vec![
                    BlobMetadata {
                        blob_header: "invalid".into(),
                        encapsulated_key: encapsulated_key_1,
                        encrypted_symmetric_key: encrypted_symmetric_key_1,
                        recipient_nonce: "nonce_1".into()
                    },
                    BlobMetadata {
                        blob_header: "invalid".into(),
                        encapsulated_key: encapsulated_key_2,
                        encrypted_symmetric_key: encrypted_symmetric_key_2,
                        recipient_nonce: "nonce_2".into()
                    },
                ],
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "no access_policy_node_id found."
        );
    }

    #[test]
    fn test_authorize_access_invalid_access_policy_node_id() {
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

        // Construct client messages.
        let blob_header_1 = BlobHeader {
            blob_id: "blob-id-1".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            access_policy_node_id: 0,
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key_1, encrypted_symmetric_key_1) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header_1).unwrap();
        let blob_header_2 = BlobHeader {
            blob_id: "blob-id-2".into(),
            key_id: cose_key.key_id.clone(),
            access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
            access_policy_node_id: 1,
            ..Default::default()
        }
        .encode_to_vec();
        let (_, encapsulated_key_2, encrypted_symmetric_key_2) =
            cfc_crypto::encrypt_message(b"plaintext", &cose_key, &blob_header_2).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata: vec![
                    BlobMetadata {
                        blob_header: blob_header_1,
                        encapsulated_key: encapsulated_key_1,
                        encrypted_symmetric_key: encrypted_symmetric_key_1,
                        recipient_nonce: "nonce_1".into()
                    },
                    BlobMetadata {
                        blob_header: blob_header_2,
                        encapsulated_key: encapsulated_key_2,
                        encrypted_symmetric_key: encrypted_symmetric_key_2,
                        recipient_nonce: "nonce_2".into()
                    },
                ],
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "access_policy_node_id is not same for all blobs in the range"
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
    fn test_authorize_access_invalid_access_policy_more_than_one_logical_pipeline() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        let recipient_tag_1 = "tag1";
        let recipient_tag_2 = "tag2";

        let access_policy = DataAccessPolicy {
            pipelines: {
                BTreeMap::from([
                    (
                        "LOGICAL_PIPELINE_1".into(),
                        LogicalPipelinePolicy {
                            instances: vec![PipelineVariantPolicy {
                                transforms: vec![pipeline_variant_policy::Transform {
                                    src_node_ids: vec![0],
                                    application: Some(ApplicationMatcher {
                                        tag: Some(recipient_tag_1.to_owned()),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ),
                    (
                        "LOGICAL_PIPELINE_2".into(),
                        LogicalPipelinePolicy {
                            instances: vec![PipelineVariantPolicy {
                                transforms: vec![pipeline_variant_policy::Transform {
                                    src_node_ids: vec![0],
                                    application: Some(ApplicationMatcher {
                                        tag: Some(recipient_tag_2.to_owned()),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ),
                ])
            },
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
            "DataAccessPolicy must have a single LogicalPipelinePolicy."
        );
    }

    #[test]
    fn test_authorize_access_invalid_access_policy_more_than_one_pipeline_variant() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        let recipient_tag_1 = "tag1";
        let recipient_tag_2 = "tag2";

        let access_policy = DataAccessPolicy {
            pipelines: {
                BTreeMap::from([(
                    "LOGICAL_PIPELINE".into(),
                    LogicalPipelinePolicy {
                        instances: vec![
                            PipelineVariantPolicy {
                                transforms: vec![pipeline_variant_policy::Transform {
                                    src_node_ids: vec![0],
                                    application: Some(ApplicationMatcher {
                                        tag: Some(recipient_tag_1.to_owned()),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            },
                            PipelineVariantPolicy {
                                transforms: vec![pipeline_variant_policy::Transform {
                                    src_node_ids: vec![0],
                                    application: Some(ApplicationMatcher {
                                        tag: Some(recipient_tag_2.to_owned()),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                )])
            },
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
            "The LogicalPipelinePolicy must have a single PipelineVariantPolicy."
        );
    }

    #[test]
    fn test_authorize_access_invalid_access_policy_no_pipeline_variants() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        let access_policy = DataAccessPolicy {
            pipelines: {
                BTreeMap::from([(
                    "LOGICAL_PIPELINE".to_owned(),
                    LogicalPipelinePolicy {
                        instances: vec![], // Empty vector for no variants
                        ..Default::default()
                    },
                )])
            },
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
            "The LogicalPipelinePolicy must have a single PipelineVariantPolicy."
        );
    }

    #[test]
    fn test_legacy_authorizes_access_but_pipline_does_not_is_disallowed() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        let legacy_policy_tag = "should_fail";
        let modern_policy_tag = "should_succeed";
        // Legacy fields in access policy grants access to the used tag, but
        // the modern policy does not so we should fail.
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(legacy_policy_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            pipelines: {
                BTreeMap::from([(
                    "LOGICAL_PIPELINE".into(),
                    LogicalPipelinePolicy {
                        instances: vec![PipelineVariantPolicy {
                            transforms: vec![pipeline_variant_policy::Transform {
                                src_node_ids: vec![0],
                                application: Some(ApplicationMatcher {
                                    tag: Some(modern_policy_tag.to_owned()),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                )])
            },
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
        let (_ciphertext, encapsulated_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

        // Request access.
        assert_err!(
            ledger.authorize_access(AuthorizeAccessRequest {
                access_policy: access_policy.to_vec(),
                blob_header: blob_header,
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: legacy_policy_tag.into(),
                recipient_nonce: "nonce".into(),
                ..Default::default()
            }),
            micro_rpc::StatusCode::FailedPrecondition,
            "requesting application does not match the access policy"
        );
    }

    #[test]
    fn test_legacy_does_not_authorize_access_but_pipline_does_is_allowed() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();

        let legacy_policy_tag = "should_fail";
        let modern_policy_tag = "should_succeed";
        // Legacy fields in access policy does not grant access to the used tag,
        // but the modern policy does so we should pass.
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(legacy_policy_tag.to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            pipelines: {
                BTreeMap::from([(
                    "LOGICAL_PIPELINE".into(),
                    LogicalPipelinePolicy {
                        instances: vec![PipelineVariantPolicy {
                            transforms: vec![pipeline_variant_policy::Transform {
                                src_node_ids: vec![0],
                                application: Some(ApplicationMatcher {
                                    tag: Some(modern_policy_tag.to_owned()),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                )])
            },
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
                recipient_tag: modern_policy_tag.into(),
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
    fn test_authorize_access_updates_budget_single_blob() {
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
            micro_rpc::StatusCode::PermissionDenied,
            ""
        );
    }

    #[test]
    fn test_authorize_access_updates_budget_multiple_blobs() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct multiple client messages
        let plaintext = b"plaintext";
        let mut blob_metadata = Vec::with_capacity(4);
        let recipient_nonce: &[u8] = b"nonce";
        for i in 0..4 {
            let blob_header = BlobHeader {
                blob_id: BlobId::from(i as u128).to_vec(),
                key_id: cose_key.key_id.clone(),
                access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                ..Default::default()
            }
            .encode_to_vec();

            let (_, encapsulated_key, encrypted_symmetric_key) =
                cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

            blob_metadata.push(BlobMetadata {
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_nonce: recipient_nonce.to_vec(),
            });
        }

        // Request access.
        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");

        // The first access should succeed.
        let response_1 = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy: access_policy.clone(),
                recipient_public_key: create_recipient_cwt(recipient_public_key.clone()),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata: blob_metadata.clone(),
                blob_range: Some(Range {
                    start: BlobId::from(0).to_vec(),
                    end: BlobId::from(4).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(response_1.reencryption_public_key, public_key);
        let authorized_blobs_1 = response_1.authorized_blob_keys;
        for i in 0..4 {
            assert_eq!(
                authorized_blobs_1.get(i).as_ref().unwrap().status.as_ref().unwrap().code,
                micro_rpc::StatusCode::Ok as i32,
            );
        }

        // But the second should fail because the budget has been exhausted.
        let response_2 = ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                recipient_public_key: create_recipient_cwt(recipient_public_key),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata,
                blob_range: Some(Range {
                    start: BlobId::from(0).to_vec(),
                    end: BlobId::from(4).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(response_2.reencryption_public_key, Vec::<u8>::new());
        let authorized_blobs_2 = response_2.authorized_blob_keys;
        for i in 0..4 {
            assert_eq!(
                authorized_blobs_2.get(i).as_ref().unwrap().status.as_ref().unwrap().code,
                micro_rpc::StatusCode::PermissionDenied as i32,
            );
        }
    }

    #[test]
    fn test_authorize_access_updates_budget_event_single_blob_event_no_rewrap_keys() {
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

        // Create the event.
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

        // Apply the event.
        let authorized_access_response =
            ledger.apply_authorize_access_event(authorize_access_event.clone(), false).unwrap();

        assert_eq!(authorized_access_response, AuthorizeAccessResponse::default());

        // The second should fail because the budget has been exhausted.
        assert_err!(
            ledger.apply_authorize_access_event(authorize_access_event, false),
            micro_rpc::StatusCode::PermissionDenied,
            ""
        );
    }

    #[test]
    fn test_authorize_access_updates_budget_multiple_blobs_event_no_rewrap_keys() {
        let (mut ledger, public_key) = create_ledger_service();
        let cose_key = extract_key_from_cwt(&public_key).unwrap();
        let recipient_tag = "tag";
        let access_policy = DataAccessPolicy {
            transforms: vec![Transform {
                application: Some(ApplicationMatcher {
                    tag: Some(recipient_tag.to_owned()),
                    ..Default::default()
                }),
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        }
        .encode_to_vec();

        // Construct multiple client messages
        let plaintext = b"plaintext";
        let mut blob_metadata = Vec::with_capacity(4);
        let recipient_nonce: &[u8] = b"nonce";
        for i in 0..4 {
            let blob_header = BlobHeader {
                blob_id: BlobId::from(i as u128).to_vec(),
                key_id: cose_key.key_id.clone(),
                access_policy_sha256: Sha256::digest(&access_policy).to_vec(),
                ..Default::default()
            }
            .encode_to_vec();

            let (_, encapsulated_key, encrypted_symmetric_key) =
                cfc_crypto::encrypt_message(plaintext, &cose_key, &blob_header).unwrap();

            blob_metadata.push(BlobMetadata {
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_nonce: recipient_nonce.to_vec(),
            });
        }

        let (_, recipient_public_key) = cfc_crypto::gen_keypair(b"key-id");

        // Create the event.
        let authorize_access_event = ledger
            .attest_and_produce_authorize_access_event(AuthorizeAccessRequest {
                access_policy: access_policy.clone(),
                recipient_public_key: create_recipient_cwt(recipient_public_key.clone()),
                recipient_tag: recipient_tag.to_owned(),
                blob_metadata: blob_metadata.clone(),
                blob_range: Some(Range {
                    start: BlobId::from(0).to_vec(),
                    end: BlobId::from(4).to_vec(),
                }),
                ..Default::default()
            })
            .unwrap();

        // Apply the event.
        let response_1 =
            ledger.apply_authorize_access_event(authorize_access_event.clone(), false).unwrap();
        assert_eq!(response_1, AuthorizeAccessResponse::default());
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
            ledger.apply_authorize_access_event(authorize_access_event, true),
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
        assert!(ledger
            .authorize_access(AuthorizeAccessRequest {
                access_policy,
                blob_header: blob_header.clone(),
                encapsulated_key,
                encrypted_symmetric_key,
                recipient_public_key: create_recipient_cwt(cfc_crypto::gen_keypair(b"key-id").1),
                recipient_tag: "tag".to_owned(),
                recipient_nonce: b"nonce2".to_vec(),
                ..Default::default()
            })
            .is_ok());

        // Now applying the event for the first request must fail.
        assert_err!(
            ledger.apply_authorize_access_event(authorize_access_event, true),
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
                            // Empty `transform_access_budgets` since the budget is unlimited.
                            transform_access_budgets: vec![RangeBudgetSnapshot {
                                start: vec![],
                                end: vec![],
                                remaining_budget: vec![],
                                default_budget: None
                            }],
                            ..Default::default()
                        }],
                        consumed_budgets: vec![],
                    }),
                }],
            }
        );
    }

    #[test]
    fn test_load_snapshot() {
        // Blob IDs have to be 16 byte long in this test to avoid failing the test
        // due to zero padding when saving the snapshot.
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
                            transform_access_budgets: vec![RangeBudgetSnapshot {
                                start: vec![b"_____blob_____1_".to_vec()],
                                end: vec![b"_____blob_____2_".to_vec()],
                                remaining_budget: vec![2],
                                default_budget: Some(50),
                            }],
                            ..Default::default()
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
                        consumed_budgets: vec![b"_____blob_____3_".to_vec()],
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
                        transform_access_budgets: vec![RangeBudgetSnapshot {
                            start: vec![],
                            end: vec![],
                            remaining_budget: vec![],
                            default_budget: None,
                        }],
                        ..Default::default()
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
