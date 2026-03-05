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

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context};
use attestation_transparency_service_proto::{
    fcp::confidentialcompute::{
        attestation_transparency_service_server, create_signing_key_request,
        create_signing_key_request::CommitKey, create_signing_key_response,
        create_signing_key_response::UnpublishedKey, CreateSigningKeyRequest,
        CreateSigningKeyResponse, GetStatusRequest, GetStatusResponse,
    },
    payload_transparency_proto::{
        fcp::confidentialcompute::{
            signed_payload::{signature, signature::Headers, Signature},
            SignedPayload,
        },
        key_proto::fcp::confidentialcompute::{key, Key},
    },
    session_proto::oak::session::v1::{SessionRequest, SessionResponse},
};
use bssl_crypto::{digest, ec, ecdsa};
use integer_encoding::VarInt;
use oak_proto_rust::oak::session::v1::EndorsedEvidence;
use oak_sdk_containers::Signer;
use payload_signer::PayloadSigner;
use prost::Message;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response};

struct KeyPair {
    signing_key: ecdsa::PrivateKey<ec::P256>,
    verifying_key: SignedPayload,
}

/// An implementation of the AttestationTransparencyService proto service.
pub struct AttestationTransparencyService<S> {
    signing_key_pair: Arc<Mutex<Option<Arc<KeyPair>>>>,
    oak_signer: Arc<S>,
    endorsed_evidence: Vec<u8>,
}

impl<S> AttestationTransparencyService<S> {
    /// Creates a new `AttestationTransparencyService`.
    pub fn create(oak_signer: S, endorsed_evidence: &EndorsedEvidence) -> anyhow::Result<Self> {
        Ok(Self {
            signing_key_pair: Default::default(),
            oak_signer: Arc::new(oak_signer),
            endorsed_evidence: endorsed_evidence.encode_to_vec(),
        })
    }

    /// Creates and returns a `PayloadSigner` that signs using the most recently
    /// created or loaded signing key (if any).
    pub fn signer(&self) -> impl PayloadSigner + Clone {
        AttestationTransparencyServiceSigner { signing_key_pair: self.signing_key_pair.clone() }
    }

    /// Converts an anyhow::Error to a tonic::Status using the attached Code (if
    /// any).
    fn convert_error(err: impl Into<anyhow::Error>) -> tonic::Status {
        let err: anyhow::Error = err.into();
        tonic::Status::new(
            err.downcast_ref::<Code>().copied().unwrap_or(Code::Internal),
            format!("{err:#}"),
        )
    }
}

#[tonic::async_trait]
impl<S: Signer + Send + Sync + 'static>
    attestation_transparency_service_server::AttestationTransparencyService
    for AttestationTransparencyService<S>
{
    type CreateSigningKeyStream = ReceiverStream<Result<CreateSigningKeyResponse, tonic::Status>>;
    type LoadSigningKeyStream = ReceiverStream<Result<SessionRequest, tonic::Status>>;
    type ShareSigningKeyStream = ReceiverStream<Result<SessionResponse, tonic::Status>>;

    async fn create_signing_key(
        &self,
        request: Request<tonic::Streaming<CreateSigningKeyRequest>>,
    ) -> Result<Response<Self::CreateSigningKeyStream>, tonic::Status> {
        let mut in_stream = request.into_inner();
        let key_pair = self.signing_key_pair.clone();
        let oak_signer = self.oak_signer.clone();
        let endorsed_evidence = self.endorsed_evidence.clone();
        // Create a tokio task that will process the request stream and write
        // the responses to a channel. See
        // https://github.com/hyperium/tonic/blob/master/examples/src/streaming/server.rs.
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let result = async {
                let request = in_stream
                    .message()
                    .await?
                    .context("CreateSigningKey aborted")
                    .context(Code::Cancelled)?;
                if !matches!(request.kind, Some(create_signing_key_request::Kind::CreateKey(_))) {
                    return Err(anyhow!("first CreateSigningKeyRequest must contain CreateKey"))
                        .context(Code::InvalidArgument);
                }

                let signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
                let serialized_verifying_key = Key {
                    algorithm: key::Algorithm::EcdsaP256.into(),
                    purpose: Some(key::Purpose::Verify.into()),
                    key_id: rand::random::<u32>().to_le_bytes().to_vec(),
                    key_material: signing_key
                        .to_public_key()
                        .to_x962_uncompressed()
                        .as_ref()
                        .to_vec(),
                }
                .encode_to_vec();

                let headers = Headers {
                    algorithm: key::Algorithm::EcdsaP256 as i32,
                    endorsed_evidence_sha256: digest::Sha256::hash(&endorsed_evidence).to_vec(),
                    ..Default::default()
                }
                .encode_to_vec();
                let signature = oak_signer
                    .sign(&build_signed_payload_sig_structure(&headers, &serialized_verifying_key))
                    .await
                    .context("failed to sign verifying key")?;

                tx.send(Ok(CreateSigningKeyResponse {
                    kind: Some(create_signing_key_response::Kind::UnpublishedKey(UnpublishedKey {
                        verifying_key: serialized_verifying_key.clone(),
                        signature: Some(Signature {
                            headers,
                            signature: Some(signature::Signature::RawSignature(
                                signature.signature,
                            )),
                            ..Default::default()
                        }),
                        endorsed_evidence,
                    })),
                }))
                .await?;

                let request = in_stream
                    .message()
                    .await?
                    .context("CreateSigningKey aborted")
                    .context(Code::Cancelled)?;
                let verifying_key = match request {
                    CreateSigningKeyRequest {
                        kind:
                            Some(create_signing_key_request::Kind::CommitKey(CommitKey {
                                verifying_key: Some(vk),
                                ..
                            })),
                        ..
                    } if vk.payload == serialized_verifying_key => vk,
                    _ => {
                        return Err(anyhow!("commit_key.verifying_key payload does not match")
                            .context(Code::InvalidArgument));
                    }
                };
                *key_pair.lock().unwrap() = Some(Arc::new(KeyPair { signing_key, verifying_key }));
                Ok::<_, anyhow::Error>(())
            };
            if let Err(err) = result.await {
                let _ = tx.send(Err(Self::convert_error(err))).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
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
        Ok(Response::new(GetStatusResponse {
            // TODO: b/454946443 - Populate version_fprint.
            version_fprint: vec![],
            verifying_key: self
                .signing_key_pair
                .lock()
                .unwrap()
                .as_ref()
                .map(|kp| kp.verifying_key.clone()),
        }))
    }
}

/// A PayloadSigner that uses the AttestationTransparencyService's most recently
/// created/loaded signing key.
#[derive(Clone)]
struct AttestationTransparencyServiceSigner {
    signing_key_pair: Arc<Mutex<Option<Arc<KeyPair>>>>,
}

impl PayloadSigner for AttestationTransparencyServiceSigner {
    fn sign(&self, headers: &[u8], payload: &[u8]) -> anyhow::Result<Signature> {
        let key_pair = self
            .signing_key_pair
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| {
                anyhow!("AttestationTransparencyService not initialized").context(Code::Unavailable)
            })?
            .clone();

        // Add the signing key's algorithm to the serialized headers. This
        // avoids mutating `headers` and is valid because concatenation of
        // serialized messages is equivalent to merging them.
        let alg_header =
            Headers { algorithm: key::Algorithm::EcdsaP256 as i32, ..Default::default() }
                .encode_to_vec();
        let serialized_headers = [headers, &alg_header].concat();
        let signature = key_pair
            .signing_key
            .sign_p1363(&build_signed_payload_sig_structure(&serialized_headers, payload));

        Ok(Signature {
            headers: serialized_headers,
            signature: Some(signature::Signature::RawSignature(signature)),
            verifier: Some(signature::Verifier::VerifyingKey(key_pair.verifying_key.clone())),
        })
    }
}

/// Constructs the message to be signed for a SignedPayload.
pub fn build_signed_payload_sig_structure(headers: &[u8], payload: &[u8]) -> Vec<u8> {
    [
        b"\x0dSignedPayload",
        headers.len().encode_var_vec().as_slice(),
        headers,
        payload.len().encode_var_vec().as_slice(),
        payload,
    ]
    .concat()
}
