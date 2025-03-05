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

use kms_proto::fcp::confidentialcompute::{
    key_management_service_server, AuthorizeConfidentialTransformRequest,
    AuthorizeConfidentialTransformResponse, ClusterPublicKey, DeriveKeysRequest,
    DeriveKeysResponse, GetClusterPublicKeyRequest, GetKeysetRequest,
    GetLogicalPipelineStateRequest, Keyset, LogicalPipelineState,
    RegisterPipelineInvocationRequest, RegisterPipelineInvocationResponse, ReleaseResultsRequest,
    ReleaseResultsResponse, RotateKeysetRequest, RotateKeysetResponse,
};
use storage_client::StorageClient;
use storage_proto::confidential_federated_compute::kms::UpdateRequest;
use tonic::{Request, Response};

/// An implementation of the KeyManagementService proto service.
pub struct KeyManagementService {
    _storage_client: StorageClient,
}

impl KeyManagementService {
    /// Creates a new KeyManagementService that interacts with persistent
    /// storage via the provided client.
    pub fn new(storage_client: StorageClient) -> Self {
        Self { _storage_client: storage_client }
    }

    /// Returns an UpdateRequest for initializing the storage.
    pub fn get_init_request() -> UpdateRequest {
        todo!()
    }
}

#[tonic::async_trait]
impl key_management_service_server::KeyManagementService for KeyManagementService {
    async fn get_cluster_public_key(
        &self,
        _request: Request<GetClusterPublicKeyRequest>,
    ) -> Result<Response<ClusterPublicKey>, tonic::Status> {
        todo!()
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
