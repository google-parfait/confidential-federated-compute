// Copyright 2025 The Trusted Computations Platform Authors.
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

use access_policies::{authorize_transform, validate_pipeline_invocation_policies};
use anyhow::{anyhow, ensure, Context};
use bssl_crypto::{digest::Sha256, ec, ecdsa};
use coset::{
    cwt::{ClaimsSetBuilder, Timestamp as CwtTimestamp},
    iana::{Algorithm, EllipticCurve},
    CborSerializable, CoseKeyBuilder,
};
use key_derivation::{
    derive_private_keys, derive_public_cwts, derive_public_keys, HPKE_BASE_X25519_SHA256_AES128GCM,
};
use kms_proto::fcp::confidentialcompute::{
    authorize_confidential_transform_response, key_management_service_server, keyset,
    AuthorizeConfidentialTransformRequest, AuthorizeConfidentialTransformResponse,
    ClusterPublicKey, DeriveKeysRequest, DeriveKeysResponse, GetClusterPublicKeyRequest,
    GetKeysetRequest, GetLogicalPipelineStateRequest, Keyset, LogicalPipelineState,
    RegisterPipelineInvocationRequest, RegisterPipelineInvocationResponse, ReleaseResultsRequest,
    ReleaseResultsResponse, RotateKeysetRequest, RotateKeysetResponse,
};
use log::warn;
use oak_crypto::encryptor::ClientEncryptor;
use oak_sdk_containers::Signer;
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use storage_client::StorageClient;
use storage_proto::{
    confidential_federated_compute::kms::{
        read_request, read_response, update_request, ClusterKeyValue, KeysetKeyValue,
        LogicalPipelineStateValue, PipelineInvocationStateValue, ReadRequest, UpdateRequest,
    },
    duration_proto::google::protobuf::Duration,
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::{Code, Request, Response};

/// Key id prefix for intermediate keys.
const INTERMEDIATE_KEY_ID: &[u8] = b"intr";

/// Returns an UpdateRequest for initializing the storage.
pub fn get_init_request() -> UpdateRequest {
    let value = ClusterKeyValue {
        key: ecdsa::PrivateKey::<ec::P256>::generate().to_big_endian().as_ref().to_vec(),
    };
    UpdateRequest {
        updates: vec![update_request::Update {
            key: StorageKey::ClusterKey.try_into().unwrap(),
            value: Some(value.encode_to_vec()),
            preconditions: Some(update_request::Preconditions {
                exists: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        }],
    }
}

/// An implementation of the KeyManagementService proto service.
pub struct KeyManagementService<SC, S> {
    storage_client: SC,
    signer: S,
}

impl<SC: StorageClient, S: Signer> KeyManagementService<SC, S> {
    /// Creates a new KeyManagementService that interacts with persistent
    /// storage via the provided client.
    pub fn new(storage_client: SC, signer: S) -> Self {
        Self { storage_client, signer }
    }

    /// Converts an anyhow::Error to a tonic::Status using the attached Code (if
    /// any).
    fn convert_error(err: anyhow::Error) -> tonic::Status {
        tonic::Status::new(
            err.downcast_ref::<Code>().copied().unwrap_or(Code::Internal),
            format!("{err:#}"),
        )
    }

    /// Returns the cluster's signing key.
    fn decode_cluster_key(
        entries: &[read_response::Entry],
    ) -> anyhow::Result<ecdsa::PrivateKey<ec::P256>> {
        let key: Vec<u8> = StorageKey::ClusterKey.try_into().unwrap();
        let entry = entries
            .iter()
            .find(|entry| entry.key == key)
            .ok_or_else(|| anyhow::Error::msg("cluster key not found"))?;
        let cluster_key = ClusterKeyValue::decode(entry.value.as_slice())?;
        ecdsa::PrivateKey::from_big_endian(cluster_key.key.as_slice())
            .ok_or_else(|| anyhow::Error::msg("failed to parse cluster key"))
    }

    /// Encodes the cluster's signing key as a COSE key.
    fn build_cluster_cose_key(key: &ecdsa::PublicKey<ec::P256>) -> Vec<u8> {
        let encoded_point = key.to_x962_uncompressed();
        // Uncompressed X9.62 starts with 0x04, then contains the x and y coordinates.
        // For P256, each coordinate is 32 bytes.
        let x = &encoded_point.as_ref()[1..33];
        let y = &encoded_point.as_ref()[33..];
        CoseKeyBuilder::new_ec2_pub_key(EllipticCurve::P_256, x.into(), y.into())
            .algorithm(Algorithm::ES256)
            // The key id is not required to be unique, so we use the beginning of the x coord.
            .key_id(x[0..4].into())
            .build()
            .to_vec()
            .unwrap()
    }

    /// Returns the creation time for a keyset key, derived from the expiration
    /// time and TTL.
    fn get_key_creation_time(ttl: &Option<Duration>, expiration: &Option<Timestamp>) -> Timestamp {
        let ttl = ttl
            .as_ref()
            .map(|ttl| {
                std::time::Duration::new(
                    ttl.seconds.try_into().unwrap_or(0),
                    ttl.nanos.try_into().unwrap_or(0),
                )
            })
            .unwrap_or(std::time::Duration::from_secs(0));
        let expiration = expiration
            .as_ref()
            .map(|exp| {
                std::time::Duration::new(
                    exp.seconds.try_into().unwrap_or(0),
                    exp.nanos.try_into().unwrap_or(0),
                )
            })
            .unwrap_or(std::time::Duration::from_secs(0));
        let created = expiration.saturating_sub(ttl);
        Timestamp { seconds: created.as_secs() as i64, nanos: created.subsec_nanos() as i32 }
    }

    /// Creates a new Range for reading from storage.
    fn create_range(
        start: StorageKey,
        end: Option<StorageKey>,
    ) -> anyhow::Result<read_request::Range> {
        Ok(read_request::Range {
            start: start.try_into()?,
            end: end.map(TryInto::try_into).transpose()?,
        })
    }

    /// Constructs the StorageKey::LogicalPipelineState for the specified
    /// logical pipeline name.
    fn get_logical_pipeline_storage_key(name: &str) -> StorageKey {
        let mut id = [0; 16];
        id.copy_from_slice(&Sha256::hash(name.as_bytes())[..16]);
        // The most significant bit of the first byte must be 1.
        id[0] |= 0x80;
        StorageKey::LogicalPipelineState { id }
    }

    /// Derives a transform's decryption keys (`src_node_ids`) and result
    /// encryption keys (`dst_node_ids`).
    async fn derive_transform_keys(
        &self,
        state: &PipelineInvocationStateValue,
        state_expiration: &Option<Timestamp>,
        src_node_ids: &[u32],
        dst_node_ids: &[u32],
    ) -> anyhow::Result<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
        let intermediates_key = state
            .intermediates_key
            .as_ref()
            .ok_or_else(|| anyhow!("PipelineInvocationState missing intermediates_key"))?;

        let mut decryption_keys = derive_private_keys(
            intermediates_key.algorithm,
            INTERMEDIATE_KEY_ID,
            &intermediates_key.ikm,
            &src_node_ids
                .iter()
                .filter(|id| **id != 0)
                .map(|id| id.to_be_bytes())
                .collect::<Vec<_>>(),
        )
        .context("failed to derive decryption keys")?;
        // If the transform reads from initial uploads (src 0), then we need to
        // provide non-intermediate keys.
        if src_node_ids.contains(&0) {
            let ranges = state
                .keyset_ids
                .iter()
                .map(|id| {
                    Self::create_range(
                        StorageKey::KeysetKey { keyset_id: *id, key_id: [0; 4] },
                        Some(StorageKey::KeysetKey { keyset_id: *id, key_id: [255; 4] }),
                    )
                    .unwrap()
                })
                .collect();
            let response = self.storage_client.read(ReadRequest { ranges }).await?;
            for entry in response.entries {
                // Skip entries that expire before the pipeline invocation
                // expires. None means that the entity doesn't expire.
                match (&entry.expiration, &state_expiration) {
                    (Some(a), Some(b)) if (a.seconds, a.nanos) < (b.seconds, b.nanos) => continue,
                    (Some(_), None) => continue,
                    _ => {}
                }

                let key_id = match entry.key.as_slice().try_into() {
                    Ok(StorageKey::KeysetKey { key_id, .. }) => key_id,
                    _ => unreachable!(),
                };
                let key = match KeysetKeyValue::decode(entry.value.as_slice()) {
                    Ok(key) => key,
                    Err(err) => {
                        // Skip keys that cannot be decoded.
                        warn!("failed to decode keyset key: {:?}", err);
                        continue;
                    }
                };
                decryption_keys.append(
                    &mut derive_private_keys(
                        key.algorithm,
                        &key_id,
                        &key.ikm,
                        &state.authorized_logical_pipeline_policies_hashes,
                    )
                    .context("failed to derive decryption keys")?,
                );
            }
        }

        let result_encryption_keys = derive_public_keys(
            intermediates_key.algorithm,
            INTERMEDIATE_KEY_ID,
            &intermediates_key.ikm,
            &dst_node_ids.iter().map(|id| id.to_be_bytes()).collect::<Vec<_>>(),
        )
        .context("failed to derive result encryption keys")?;

        Ok((decryption_keys, result_encryption_keys))
    }
}

#[tonic::async_trait]
impl<SC, S> key_management_service_server::KeyManagementService for KeyManagementService<SC, S>
where
    SC: StorageClient + Send + Sync + 'static,
    S: Signer + Send + Sync + 'static,
{
    async fn get_cluster_public_key(
        &self,
        _request: Request<GetClusterPublicKeyRequest>,
    ) -> Result<Response<ClusterPublicKey>, tonic::Status> {
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![Self::create_range(StorageKey::ClusterKey, None).unwrap()],
            })
            .await
            .map_err(Self::convert_error)?;
        let cluster_key =
            Self::decode_cluster_key(&response.entries).map_err(Self::convert_error)?;
        let public_key = Self::build_cluster_cose_key(&cluster_key.to_public_key());
        Ok(Response::new(ClusterPublicKey { public_key }))
    }

    async fn get_keyset(
        &self,
        request: Request<GetKeysetRequest>,
    ) -> Result<Response<Keyset>, tonic::Status> {
        // Read all potential keys in the keyset.
        let request = request.into_inner();
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![Self::create_range(
                    StorageKey::KeysetKey { keyset_id: request.keyset_id, key_id: [0; 4] },
                    Some(StorageKey::KeysetKey { keyset_id: request.keyset_id, key_id: [255; 4] }),
                )
                .unwrap()],
            })
            .await
            .map_err(Self::convert_error)?;

        let keys: Vec<_> = response
            .entries
            .into_iter()
            .filter_map(|entry| {
                let key_id = match entry.key.as_slice().try_into() {
                    Ok(StorageKey::KeysetKey { key_id, .. }) => key_id,
                    _ => unreachable!(),
                };
                // Skip keys that cannot be decoded.
                let value = KeysetKeyValue::decode(entry.value.as_slice()).ok()?;
                Some(keyset::Key {
                    key_id: key_id.into(),
                    created: Some(Self::get_key_creation_time(&value.ttl, &entry.expiration)),
                    expiration: entry.expiration,
                })
            })
            .collect();
        if keys.is_empty() {
            Err(tonic::Status::not_found("no keys found in keyset"))
        } else {
            Ok(Response::new(Keyset { keyset_id: request.keyset_id, keys }))
        }
    }

    async fn rotate_keyset(
        &self,
        request: Request<RotateKeysetRequest>,
    ) -> Result<Response<RotateKeysetResponse>, tonic::Status> {
        let request = request.into_inner();
        let ttl = request.ttl.unwrap_or_default();
        let value = KeysetKeyValue {
            ikm: bssl_crypto::rand_array::<32>().into(),
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ttl: Some(ttl.clone()),
        }
        .encode_to_vec();

        // Create a new key with a random key id. The key id isn't sensitive, so
        // we don't need a secure random number generator. In case of collision
        // (FAILED_PRECONDITION), try again with quadratic probing.
        let key_id: u32 = rand::random();
        for i in 0..10 {
            let result = self
                .storage_client
                .update(UpdateRequest {
                    updates: vec![update_request::Update {
                        key: StorageKey::KeysetKey {
                            keyset_id: request.keyset_id,
                            key_id: key_id.wrapping_add(i * i).to_be_bytes(),
                        }
                        .try_into()
                        .unwrap(),
                        value: Some(value.clone()),
                        ttl: Some(ttl.clone()),
                        preconditions: Some(update_request::Preconditions {
                            exists: Some(false),
                            ..Default::default()
                        }),
                    }],
                })
                .await;
            match result {
                Ok(_) => return Ok(Response::new(RotateKeysetResponse::default())),
                Err(err) if err.downcast_ref::<Code>() == Some(&Code::FailedPrecondition) => {
                    // The `exists: false` precondition failed, so the key
                    // already exists. Try again with a different key id.
                    continue;
                }
                Err(err) => return Err(Self::convert_error(err)),
            }
        }
        Err(tonic::Status::unavailable("failed to find an unused key id"))
    }

    async fn derive_keys(
        &self,
        request: Request<DeriveKeysRequest>,
    ) -> Result<Response<DeriveKeysResponse>, tonic::Status> {
        // Read all potential keys in the keyset.
        let request = request.into_inner();
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![Self::create_range(
                    StorageKey::KeysetKey { keyset_id: request.keyset_id, key_id: [0; 4] },
                    Some(StorageKey::KeysetKey { keyset_id: request.keyset_id, key_id: [255; 4] }),
                )
                .unwrap()],
            })
            .await
            .map_err(Self::convert_error)?;

        // Find the most recently created key.
        let (key_id, value, created, expiration) = response
            .entries
            .into_iter()
            .filter_map(|entry| {
                let key_id = match entry.key.as_slice().try_into() {
                    Ok(StorageKey::KeysetKey { key_id, .. }) => key_id,
                    _ => unreachable!(),
                };
                // Skip keys that cannot be decoded.
                let value = KeysetKeyValue::decode(entry.value.as_slice()).ok()?;
                let created = Self::get_key_creation_time(&value.ttl, &entry.expiration);
                Some((key_id, value, created, entry.expiration))
            })
            .max_by_key(|entry| (entry.2.seconds, entry.2.nanos))
            .ok_or_else(|| tonic::Status::not_found("no keys found in keyset"))?;

        let public_keys = derive_public_cwts(
            value.algorithm,
            &key_id,
            &value.ikm,
            ClaimsSetBuilder::new()
                .issued_at(CwtTimestamp::WholeSeconds(response.now.map(|t| t.seconds).unwrap_or(0)))
                .not_before(CwtTimestamp::WholeSeconds(created.seconds))
                .expiration_time(CwtTimestamp::WholeSeconds(
                    expiration.map(|t| t.seconds).unwrap_or(0),
                ))
                .build(),
            &request.authorized_logical_pipelines_hashes,
            &self.signer,
        )
        .await
        .map_err(Self::convert_error)?;
        Ok(Response::new(DeriveKeysResponse { public_keys }))
    }

    async fn get_logical_pipeline_state(
        &self,
        request: Request<GetLogicalPipelineStateRequest>,
    ) -> Result<Response<LogicalPipelineState>, tonic::Status> {
        let request = request.into_inner();
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![Self::create_range(
                    Self::get_logical_pipeline_storage_key(&request.name),
                    None,
                )
                .unwrap()],
            })
            .await
            .map_err(Self::convert_error)?;
        let entry = response
            .entries
            .first()
            .ok_or_else(|| tonic::Status::not_found("logical pipeline has no saved state"))?;
        let value = LogicalPipelineStateValue::decode(entry.value.as_slice())
            .context("failed to decode LogicalPipelineStateValue")
            .map_err(Self::convert_error)?;
        Ok(Response::new(LogicalPipelineState { name: request.name, value: value.state }))
    }

    async fn register_pipeline_invocation(
        &self,
        request: Request<RegisterPipelineInvocationRequest>,
    ) -> Result<Response<RegisterPipelineInvocationResponse>, tonic::Status> {
        let request = request.into_inner();
        validate_pipeline_invocation_policies(
            &request.logical_pipeline_name,
            &request.pipeline_variant_policy,
            request.authorized_logical_pipeline_policies.as_slice(),
        )
        .context("invalid pipeline policies")
        .context(Code::InvalidArgument)
        .map_err(Self::convert_error)?;

        // Create a future that writes the pipeline invocation state to storage.
        // We don't worry about retring on collision because the probability is
        // so low.
        let write_invocation_state_fut = async {
            let id = rand::random();
            let ttl = request.intermediates_ttl.unwrap_or_default();
            let authorized_logical_pipeline_policies_hashes = request
                .authorized_logical_pipeline_policies
                .into_iter()
                .map(|policy| Sha256::hash(policy.as_slice()).into())
                .collect();
            let value = PipelineInvocationStateValue {
                logical_pipeline_name: request.logical_pipeline_name.clone(),
                pipeline_variant_policy_hash: Sha256::hash(&request.pipeline_variant_policy).into(),
                intermediates_key: Some(KeysetKeyValue {
                    ikm: bssl_crypto::rand_array::<32>().into(),
                    algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
                    ttl: Some(ttl.clone()),
                }),
                keyset_ids: request.keyset_ids,
                authorized_logical_pipeline_policies_hashes,
            };
            self.storage_client
                .update(UpdateRequest {
                    updates: vec![update_request::Update {
                        key: StorageKey::PipelineInvocationState { id }.try_into().unwrap(),
                        value: Some(value.encode_to_vec()),
                        ttl: Some(ttl),
                        preconditions: Some(update_request::Preconditions {
                            exists: Some(false),
                            ..Default::default()
                        }),
                    }],
                })
                .await
                .map_err(Self::convert_error)?;
            Ok(id.into())
        };

        // Create a future to retrieve the current logical pipeline state (if
        // any).
        let logical_pipeline_state_fut = async {
            let response = self
                .get_logical_pipeline_state(Request::new(GetLogicalPipelineStateRequest {
                    name: request.logical_pipeline_name.clone(),
                }))
                .await;
            match response {
                Ok(response) => Ok(Some(response.into_inner())),
                Err(err) if err.code() == Code::NotFound => Ok(None),
                Err(err) => Err(err),
            }
        };

        let (invocation_id, logical_pipeline_state) =
            tokio::try_join!(write_invocation_state_fut, logical_pipeline_state_fut)?;
        Ok(Response::new(RegisterPipelineInvocationResponse {
            invocation_id,
            logical_pipeline_state,
        }))
    }

    async fn authorize_confidential_transform(
        &self,
        request: Request<AuthorizeConfidentialTransformRequest>,
    ) -> Result<Response<AuthorizeConfidentialTransformResponse>, tonic::Status> {
        let request = request.into_inner();

        // Look up the cluster signing key and the pipeline invocation state.
        let id = request
            .invocation_id
            .as_slice()
            .try_into()
            .map_err(|_| tonic::Status::invalid_argument("invocation_id is invalid"))?;
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![
                    Self::create_range(StorageKey::ClusterKey, None).unwrap(),
                    Self::create_range(StorageKey::PipelineInvocationState { id }, None).unwrap(),
                ],
            })
            .await
            .map_err(Self::convert_error)?;
        let cluster_key =
            Self::decode_cluster_key(&response.entries).map_err(Self::convert_error)?;
        let (state, state_expiration) = response
            .entries
            .iter()
            .find_map(|entry| match entry.key.as_slice().try_into() {
                Ok(StorageKey::PipelineInvocationState { .. }) => Some(
                    PipelineInvocationStateValue::decode(entry.value.as_slice())
                        .map(|state| (state, entry.expiration.clone()))
                        .context("failed to decode PipelineInvocationStateValue")
                        .map_err(Self::convert_error),
                ),
                _ => None,
            })
            .unwrap_or_else(|| Err(tonic::Status::not_found("pipeline invocation not found")))?;
        if Sha256::hash(&request.pipeline_variant_policy) != *state.pipeline_variant_policy_hash {
            return Err(tonic::Status::invalid_argument(
                "request does not match registered pipeline variant policy",
            ));
        }

        // Check whether the transform is authorized by the policy.
        let authorized_transform = authorize_transform(
            &request.pipeline_variant_policy,
            request
                .evidence
                .as_ref()
                .ok_or_else(|| tonic::Status::invalid_argument("evidence is required"))?,
            request
                .endorsements
                .as_ref()
                .ok_or_else(|| tonic::Status::invalid_argument("endorsements are required"))?,
            &request.tag,
            &response
                .now
                .ok_or_else(|| tonic::Status::internal("StorageResponse missing timestamp"))?,
        )
        .context(Code::InvalidArgument)
        .map_err(Self::convert_error)?;

        // Generate the transform's encryption and decryption keys.
        let (decryption_keys, result_encryption_keys) = self
            .derive_transform_keys(
                &state,
                &state_expiration,
                &authorized_transform.src_node_ids,
                &authorized_transform.dst_node_ids,
            )
            .await
            .map_err(Self::convert_error)?;

        // Construct the encrypted portion of the response.
        let protected_response = authorize_confidential_transform_response::ProtectedResponse {
            decryption_keys,
            result_encryption_keys,
        };
        let associated_data = authorize_confidential_transform_response::AssociatedData {
            cluster_public_key: Self::build_cluster_cose_key(&cluster_key.to_public_key()),
            config_constraints: authorized_transform.config_constraints,
        };
        let encrypted_message =
            ClientEncryptor::create(&authorized_transform.extracted_evidence.encryption_public_key)
                .and_then(|mut encryptor| {
                    encryptor.encrypt(
                        &protected_response.encode_to_vec(),
                        &associated_data.encode_to_vec(),
                    )
                })
                .and_then(|msg| Ok(msg.convert()?))
                .context("failed to encrypt response")
                .map_err(Self::convert_error)?;

        Ok(Response::new(AuthorizeConfidentialTransformResponse {
            protected_response: Some(encrypted_message),
            // TODO: b/398874186 - Populate the signing key endorsement.
            signing_key_endorsement: vec![],
        }))
    }

    async fn release_results(
        &self,
        _request: Request<ReleaseResultsRequest>,
    ) -> Result<Response<ReleaseResultsResponse>, tonic::Status> {
        todo!()
    }
}

/// Map keys for data stored in the TCP Storage service. All values can be
/// encoded to/from 16 byte strings.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum StorageKey {
    /// Identifies the TCP cluster's signing key.
    ClusterKey,

    /// Identifies a keyset.
    KeysetKey { keyset_id: u64, key_id: [u8; 4] },

    /// Identifies a pipeline invocation.
    PipelineInvocationState { id: [u8; 12] },

    /// Identifies a logical pipeline. The most significant bit of the first
    /// byte is always 1.
    LogicalPipelineState { id: [u8; 16] },
}

impl TryFrom<&[u8]> for StorageKey {
    type Error = anyhow::Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        ensure!(value.len() == 16, "invalid StorageKey length: {}", value.len());
        match u32::from_be_bytes(value[0..4].try_into()?) {
            0 if value[4..16] == [0; 12] => Ok(StorageKey::ClusterKey),
            1 => Ok(StorageKey::KeysetKey {
                keyset_id: u64::from_be_bytes(value[4..12].try_into()?),
                key_id: value[12..16].try_into()?,
            }),
            2 => Ok(StorageKey::PipelineInvocationState { id: (&value[4..16]).try_into()? }),
            x if x >= 0x80000000 => Ok(StorageKey::LogicalPipelineState { id: value.try_into()? }),
            _ => Err(anyhow!("unsupported StorageKey encoding")),
        }
    }
}

impl TryFrom<StorageKey> for Vec<u8> {
    type Error = anyhow::Error;

    fn try_from(value: StorageKey) -> Result<Self, Self::Error> {
        match value {
            StorageKey::ClusterKey => Ok(vec![0; 16]),
            StorageKey::KeysetKey { keyset_id, key_id } => {
                let mut x = vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                x[4..12].copy_from_slice(&keyset_id.to_be_bytes());
                x[12..16].copy_from_slice(&key_id);
                Ok(x)
            }
            StorageKey::PipelineInvocationState { id } => {
                let mut x = vec![0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                x[4..16].copy_from_slice(&id);
                Ok(x)
            }
            StorageKey::LogicalPipelineState { id } => {
                anyhow::ensure!(id[0] >= 0x80, "invalid LogicalPipeline id");
                Ok(id.to_vec())
            }
        }
    }
}
