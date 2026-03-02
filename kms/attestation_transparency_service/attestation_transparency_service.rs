// Copyright 2026 Google LLC.
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

use anyhow::anyhow;
use attestation_transparency_service_proto::{
    fcp::confidentialcompute::{
        attestation_transparency_service_server, CreateSigningKeyRequest, CreateSigningKeyResponse,
        GetStatusRequest, GetStatusResponse,
    },
    payload_transparency_proto::fcp::confidentialcompute::signed_payload::Signature,
    session_proto::oak::session::v1::{SessionRequest, SessionResponse},
};
use payload_signer::PayloadSigner;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response};

/// An implementation of the AttestationTransparencyService proto service.
#[derive(Default)]
pub struct AttestationTransparencyService {}

impl AttestationTransparencyService {
    /// Creates and returns a `PayloadSigner` that signs using the most recently
    /// created or loaded signing key (if any).
    pub fn signer(&self) -> impl PayloadSigner + Clone {
        AttestationTransparencyServiceSigner {}
    }
}

#[tonic::async_trait]
impl attestation_transparency_service_server::AttestationTransparencyService
    for AttestationTransparencyService
{
    type CreateSigningKeyStream = ReceiverStream<Result<CreateSigningKeyResponse, tonic::Status>>;
    type LoadSigningKeyStream = ReceiverStream<Result<SessionRequest, tonic::Status>>;
    type ShareSigningKeyStream = ReceiverStream<Result<SessionResponse, tonic::Status>>;

    async fn create_signing_key(
        &self,
        _request: Request<tonic::Streaming<CreateSigningKeyRequest>>,
    ) -> Result<Response<Self::CreateSigningKeyStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("CreateSigningKey is unimplemented"))
    }

    async fn share_signing_key(
        &self,
        _request: Request<tonic::Streaming<SessionRequest>>,
    ) -> Result<Response<Self::ShareSigningKeyStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("ShareSigningKey is unimplemented"))
    }

    async fn load_signing_key(
        &self,
        _request: Request<tonic::Streaming<SessionResponse>>,
    ) -> Result<Response<Self::LoadSigningKeyStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("LoadSigningKey is unimplemented"))
    }

    async fn get_status(
        &self,
        _request: Request<GetStatusRequest>,
    ) -> Result<Response<GetStatusResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("GetStatus is unimplemented"))
    }
}

/// A PayloadSigner that uses the AttestationTransparencyService's most recently
/// created/loaded signing key.
#[derive(Clone)]
struct AttestationTransparencyServiceSigner {}

impl PayloadSigner for AttestationTransparencyServiceSigner {
    fn sign(&self, _headers: &[u8], _payload: &[u8]) -> anyhow::Result<Signature> {
        Err(anyhow!("AttestationTransparencyService not initialized")
            .context(tonic::Code::Unavailable))
    }
}
