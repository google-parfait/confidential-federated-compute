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

use anyhow::{anyhow, ensure};
use bssl_crypto::{ec, ecdsa};
use coset::{cbor::value::Value, iana, Algorithm, CborSerializable, CoseKey, KeyType, Label};
use kms_proto::fcp::confidentialcompute::{
    key_management_service_server, AuthorizeConfidentialTransformRequest,
    AuthorizeConfidentialTransformResponse, ClusterPublicKey, DeriveKeysRequest,
    DeriveKeysResponse, GetClusterPublicKeyRequest, GetKeysetRequest,
    GetLogicalPipelineStateRequest, Keyset, LogicalPipelineState,
    RegisterPipelineInvocationRequest, RegisterPipelineInvocationResponse, ReleaseResultsRequest,
    ReleaseResultsResponse, RotateKeysetRequest, RotateKeysetResponse,
};
use prost::Message;
use storage_client::StorageClient;
use storage_proto::confidential_federated_compute::kms::{
    read_request, update_request, ClusterKeyValue, ReadRequest, UpdateRequest,
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
pub struct KeyManagementService<SC> {
    storage_client: SC,
}

impl<SC: StorageClient> KeyManagementService<SC> {
    /// Creates a new KeyManagementService that interacts with persistent
    /// storage via the provided client.
    pub fn new(storage_client: SC) -> Self {
        Self { storage_client }
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
                ranges: vec![read_request::Range {
                    start: StorageKey::ClusterKey.try_into().unwrap(),
                    end: None,
                }],
            })
            .await?;
        ensure!(response.entries.len() == 1, "cluster key not found");
        let cluster_key = ClusterKeyValue::decode(response.entries[0].value.as_slice())?;
        ecdsa::PrivateKey::from_big_endian(cluster_key.key.as_slice())
            .ok_or_else(|| anyhow::Error::msg("failed to parse cluster key"))
    }
}

#[tonic::async_trait]
impl<SC> key_management_service_server::KeyManagementService for KeyManagementService<SC>
where
    SC: StorageClient + Send + Sync + 'static,
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
        _request: Request<GetKeysetRequest>,
    ) -> Result<Response<Keyset>, tonic::Status> {
        todo!()
    }

    async fn rotate_keyset(
        &self,
        _request: Request<RotateKeysetRequest>,
    ) -> Result<Response<RotateKeysetResponse>, tonic::Status> {
        todo!()
    }

    async fn derive_keys(
        &self,
        _request: Request<DeriveKeysRequest>,
    ) -> Result<Response<DeriveKeysResponse>, tonic::Status> {
        todo!()
    }

    async fn get_logical_pipeline_state(
        &self,
        _request: Request<GetLogicalPipelineStateRequest>,
    ) -> Result<Response<LogicalPipelineState>, tonic::Status> {
        todo!()
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
}

impl TryFrom<&[u8]> for StorageKey {
    type Error = anyhow::Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        ensure!(value.len() == 16, "invalid StorageKey length: {}", value.len());
        match u32::from_be_bytes(value[0..4].try_into()?) {
            0 if value[4..16] == [0; 12] => Ok(StorageKey::ClusterKey),
            _ => Err(anyhow!("unsupported StorageKey encoding")),
        }
    }
}

impl TryFrom<StorageKey> for Vec<u8> {
    type Error = anyhow::Error;

    fn try_from(value: StorageKey) -> Result<Self, Self::Error> {
        match value {
            StorageKey::ClusterKey => Ok(vec![0; 16]),
        }
    }
}
