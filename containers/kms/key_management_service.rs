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

use anyhow::{anyhow, ensure, Context};
use bssl_crypto::{digest::Sha256, ec, ecdsa};
use coset::{
    cbor::value::Value,
    cwt::{ClaimsSetBuilder, Timestamp as CwtTimestamp},
    iana, Algorithm, CborSerializable, CoseKey, KeyType, Label,
};
use key_derivation::{derive_public_keys, HPKE_BASE_X25519_SHA256_AES128GCM};
use kms_proto::fcp::confidentialcompute::{
    key_management_service_server, keyset, AuthorizeConfidentialTransformRequest,
    AuthorizeConfidentialTransformResponse, ClusterPublicKey, DeriveKeysRequest,
    DeriveKeysResponse, GetClusterPublicKeyRequest, GetKeysetRequest,
    GetLogicalPipelineStateRequest, Keyset, LogicalPipelineState,
    RegisterPipelineInvocationRequest, RegisterPipelineInvocationResponse, ReleaseResultsRequest,
    ReleaseResultsResponse, RotateKeysetRequest, RotateKeysetResponse,
};
use oak_sdk_containers::Signer;
use prost::Message;
use storage_client::StorageClient;
use storage_proto::{
    confidential_federated_compute::kms::{
        read_request, update_request, ClusterKeyValue, KeysetKeyValue, LogicalPipelineStateValue,
        ReadRequest, UpdateRequest,
    },
    duration_proto::google::protobuf::Duration,
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::{Code, Request, Response};

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
    async fn get_cluster_key(&self) -> anyhow::Result<ecdsa::PrivateKey<ec::P256>> {
        let response = self
            .storage_client
            .read(ReadRequest {
                ranges: vec![Self::create_range(StorageKey::ClusterKey, None).unwrap()],
            })
            .await?;
        ensure!(response.entries.len() == 1, "cluster key not found");
        let cluster_key = ClusterKeyValue::decode(response.entries[0].value.as_slice())?;
        ecdsa::PrivateKey::from_big_endian(cluster_key.key.as_slice())
            .ok_or_else(|| anyhow::Error::msg("failed to parse cluster key"))
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
        let cluster_key = self.get_cluster_key().await.map_err(Self::convert_error)?;
        let encoded_point = cluster_key.to_public_key().to_x962_uncompressed();
        // Uncompressed X9.62 starts with 0x04, then contains the x and y coordinates.
        // For P256, each coordinate is 32 bytes.
        let x = &encoded_point.as_ref()[1..33];
        let y = &encoded_point.as_ref()[33..];
        let public_key = CoseKey {
            kty: KeyType::Assigned(iana::KeyType::EC2),
            alg: Some(Algorithm::Assigned(iana::Algorithm::ES256)),
            // The key id is not required to be unique, so we use the beginning of the x coord.
            key_id: x[0..4].into(),
            params: vec![
                (
                    Label::Int(iana::Ec2KeyParameter::Crv as i64),
                    Value::from(iana::EllipticCurve::P_256 as u64),
                ),
                (Label::Int(iana::Ec2KeyParameter::X as i64), Value::Bytes(x.into())),
                (Label::Int(iana::Ec2KeyParameter::Y as i64), Value::Bytes(y.into())),
            ],
            ..Default::default()
        };
        Ok(Response::new(ClusterPublicKey { public_key: public_key.to_vec().unwrap() }))
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

        let public_keys = derive_public_keys(
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
        _request: Request<RegisterPipelineInvocationRequest>,
    ) -> Result<Response<RegisterPipelineInvocationResponse>, tonic::Status> {
        todo!()
    }

    async fn authorize_confidential_transform(
        &self,
        _request: Request<AuthorizeConfidentialTransformRequest>,
    ) -> Result<Response<AuthorizeConfidentialTransformResponse>, tonic::Status> {
        todo!()
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
            StorageKey::LogicalPipelineState { id } => {
                anyhow::ensure!(id[0] >= 0x80, "invalid LogicalPipeline id");
                Ok(id.to_vec())
            }
        }
    }
}
