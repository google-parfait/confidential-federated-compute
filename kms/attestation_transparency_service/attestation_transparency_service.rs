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

use anyhow::{anyhow, bail, ensure, Context};
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
use bssl_crypto::{digest, digest::Algorithm as _, ec, ecdsa};
use integer_encoding::VarInt;
use oak_attestation_verification::{
    AmdSevSnpDiceAttestationVerifier, AmdSevSnpPolicy, ContainerPolicy, FirmwarePolicy,
    InsecureAttestationVerifier, KernelPolicy, SystemPolicy,
};
use oak_attestation_verification_types::{policy::EventPolicy, verifier::AttestationVerifier};
use oak_proto_rust::oak::{
    attestation::v1::TeePlatform,
    session::v1::{EndorsedEvidence, PlaintextMessage},
};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::Signer;
use oak_session::{
    aggregators::PassThrough,
    attestation::AttestationType,
    config::SessionConfig,
    generator::BindableAssertionGenerator,
    handshake::HandshakeType,
    key_extractor::DefaultBindingKeyExtractor,
    session_binding::{SessionBinder, SignatureBindingVerifierProvider},
    verifier::BoundAssertionVerifier,
    ClientSession, ProtocolEngine, ServerSession, Session,
};
use oak_session_endorsed_evidence::{
    EndorsedEvidenceBindableAssertionGenerator, EndorsedEvidenceBoundAssertionVerifier,
};
use oak_time::Clock;
use payload_signer::PayloadSigner;
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use serde::{de::Error as _, Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response};

const ASSERTION_ID: &str = "cfc_ats";

#[derive(Serialize, Deserialize)]
struct KeyPair {
    #[serde(
        serialize_with = "serialize_private_key",
        deserialize_with = "deserialize_private_key"
    )]
    signing_key: ecdsa::PrivateKey<ec::P256>,

    #[serde(serialize_with = "serialize_message", deserialize_with = "deserialize_message")]
    verifying_key: SignedPayload,
}

fn serialize_private_key<S: serde::ser::Serializer, C: ec::Curve>(
    value: &ecdsa::PrivateKey<C>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(value.to_big_endian().as_ref())
}

fn deserialize_private_key<'de, D: serde::de::Deserializer<'de>, C: ec::Curve>(
    deserializer: D,
) -> Result<ecdsa::PrivateKey<C>, D::Error> {
    ecdsa::PrivateKey::from_big_endian(&Vec::<u8>::deserialize(deserializer)?)
        .ok_or_else(|| D::Error::custom("invalid private key"))
}

fn serialize_message<S: serde::ser::Serializer, M: Message>(
    value: &M,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(&value.encode_to_vec())
}

fn deserialize_message<'de, D: serde::de::Deserializer<'de>, M: Message + Default>(
    deserializer: D,
) -> Result<M, D::Error> {
    M::decode(Vec::<u8>::deserialize(deserializer)?.as_slice())
        .map_err(|err| D::Error::custom(format!("failed to deserialize message: {err}")))
}

/// Performs the initial handshake for a Session, reading messages from `rx` and
/// writing responses to `tx`.
async fn initialize_session<'a, I, O, S>(
    mut session: S,
    rx: &'a mut tonic::Streaming<I>,
    tx: &'a mpsc::Sender<Result<O, tonic::Status>>,
) -> anyhow::Result<S>
where
    I: ProstProtoConversionExt<S::Input> + 'static,
    O: Message + Default + 'static,
    S: ProtocolEngine + Session + Send + 'a + 'static,
    S::Input: Message + Default,
    S::Output: ProstProtoConversionExt<O>,
{
    while !session.is_open() {
        let in_msg = rx.message().await?.context("stream unexpectedly closed")?;
        // `oak_sdk_containers::InstanceSessionBinder` performs blocking
        // operations, so it cannot be called from an async thread. The
        // SessionBinder is only used during the initial handshake, so it's not
        // necessary to run subsequent ProtocolEngine interactions on a separate
        // thread.
        let out_msgs: Vec<O>;
        (session, out_msgs) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            session
                .put_incoming_message(in_msg.convert()?)
                .context(tonic::Code::InvalidArgument)?;
            let mut out_msgs = Vec::with_capacity(1);
            while let Some(out_msg) = session.get_outgoing_message()? {
                out_msgs.push(out_msg.convert()?);
            }
            Ok((session, out_msgs))
        })
        .await??;
        for out_msg in out_msgs {
            tx.send(Ok(out_msg)).await?;
        }
    }
    Ok(session)
}

/// An implementation of the AttestationTransparencyService proto service.
pub struct AttestationTransparencyService<S> {
    signing_key_pair: Arc<Mutex<Option<Arc<KeyPair>>>>,
    oak_signer: Arc<S>,
    endorsed_evidence: Vec<u8>,
    assertion_generator: Arc<dyn BindableAssertionGenerator>,
    assertion_verifier: Arc<dyn BoundAssertionVerifier>,
    version_fprint: Vec<u8>,
}

impl<S> AttestationTransparencyService<S> {
    /// Creates a new `AttestationTransparencyService`.
    pub fn create(
        oak_signer: S,
        endorsed_evidence: &EndorsedEvidence,
        session_binder: Arc<dyn SessionBinder>,
        clock: Arc<dyn Clock>,
    ) -> anyhow::Result<Self> {
        let evidence =
            endorsed_evidence.evidence.clone().context("EndorsedEvidence.evidence is not set")?;
        let endorsements = endorsed_evidence
            .endorsements
            .clone()
            .context("EndorsedEvidence.endorsements is not set")?;

        // Compute a fingerprint of the current software and firmware versions.
        // This is the same information that's being checked by the policies.
        // While proto serialization is not canonical, it's (very) likely to be
        // deterministic for the same version of the code -- and if the version
        // is different, the digest will be different anyway.
        let mut version_fprint = digest::Sha256::new();
        fn update_fprint<'a, D: digest::Algorithm, M: Message>(
            digest: &mut D,
            msg: &'a M,
        ) -> &'a M {
            let encoded = msg.encode_to_vec();
            digest.update(&encoded.len().to_le_bytes());
            digest.update(&encoded);
            msg
        }

        ensure!(
            evidence.event_log.as_ref().map(|el| el.encoded_events.len()).unwrap_or_default() == 3,
            "event log must contain exactly 3 events"
        );
        let event_policies: Vec<Box<dyn EventPolicy>> = vec![
            Box::new(KernelPolicy::new(update_fprint(
                &mut version_fprint,
                &KernelPolicy::evidence_to_reference_values(
                    &evidence.event_log.as_ref().unwrap().encoded_events[0],
                )
                .context("building KernelPolicy")?,
            ))),
            Box::new(SystemPolicy::new(update_fprint(
                &mut version_fprint,
                &SystemPolicy::evidence_to_reference_values(
                    &evidence.event_log.as_ref().unwrap().encoded_events[1],
                )
                .context("building SystemPolicy")?,
            ))),
            Box::new(ContainerPolicy::new(update_fprint(
                &mut version_fprint,
                &ContainerPolicy::evidence_to_reference_values(
                    &evidence.event_log.as_ref().unwrap().encoded_events[2],
                )
                .context("building ContainerPolicy")?,
            ))),
        ];
        let peer_verifier: Arc<dyn AttestationVerifier> = match evidence
            .root_layer
            .as_ref()
            .map(|rl| rl.platform.try_into())
        {
            Some(Ok(TeePlatform::AmdSevSnp)) => {
                let (amd_sev_snp_rv, firmware_rv) = AmdSevSnpPolicy::evidence_to_reference_values(
                    evidence.root_layer.as_ref().unwrap(),
                )
                .context("building AmdSevSnpPolicy")?;
                Arc::new(AmdSevSnpDiceAttestationVerifier::new(
                    AmdSevSnpPolicy::new(update_fprint(&mut version_fprint, &amd_sev_snp_rv)),
                    Box::new(FirmwarePolicy::new(update_fprint(&mut version_fprint, &firmware_rv))),
                    event_policies,
                    clock,
                ))
            }

            Some(Ok(TeePlatform::None)) => {
                Arc::new(InsecureAttestationVerifier::new(clock, event_policies))
            }

            platform => bail!("platform {platform:?} is not supported"),
        };

        Ok(Self {
            signing_key_pair: Default::default(),
            oak_signer: Arc::new(oak_signer),
            endorsed_evidence: endorsed_evidence.encode_to_vec(),
            assertion_generator: Arc::new(EndorsedEvidenceBindableAssertionGenerator::new(
                Arc::new(StaticAttester::new(evidence)),
                Arc::new(StaticEndorser::new(endorsements)),
                session_binder,
            )),
            assertion_verifier: Arc::new(EndorsedEvidenceBoundAssertionVerifier::new(
                peer_verifier,
                Arc::new(SignatureBindingVerifierProvider::new(Arc::new(
                    DefaultBindingKeyExtractor {},
                ))),
            )),
            version_fprint: version_fprint.digest_to_vec(),
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

    /// Returns an SessionConfig that accepts equivalent VMs.
    fn get_session_config(&self) -> SessionConfig {
        SessionConfig::builder(AttestationType::Bidirectional, HandshakeType::NoiseNN)
            .add_self_assertion_generator_ref(String::from(ASSERTION_ID), &self.assertion_generator)
            .add_peer_assertion_verifier_ref(String::from(ASSERTION_ID), &self.assertion_verifier)
            // Since only one assertion type is used, a trivial PassThrough
            // aggregator is sufficient.
            .set_assertion_attestation_aggregator(Box::new(PassThrough {}))
            .build()
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
        request: Request<tonic::Streaming<SessionRequest>>,
    ) -> Result<Response<Self::ShareSigningKeyStream>, tonic::Status> {
        let mut in_stream = request.into_inner();
        let key_pair =
            self.signing_key_pair.lock().unwrap().clone().ok_or_else(|| {
                tonic::Status::failed_precondition("no signing key has been created")
            })?;
        let session_config = self.get_session_config();
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let result = async {
                let mut session = ServerSession::create(session_config)
                    .context("failed to create ServerSession")?;
                session = initialize_session(session, &mut in_stream, &tx).await?;

                let mut plaintext = Vec::new();
                ciborium::into_writer(&*key_pair, &mut plaintext)
                    .context("failed to encode KeyPair")?;
                session.write(PlaintextMessage { plaintext })?;
                tx.send(Ok(session
                    .get_outgoing_message()?
                    .context("get_outgoing_message returned None")?
                    .convert()?))
                    .await?;
                Ok::<_, anyhow::Error>(())
            };
            if let Err(err) = result.await {
                let _ = tx.send(Err(Self::convert_error(err))).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn load_signing_key(
        &self,
        request: Request<tonic::Streaming<SessionResponse>>,
    ) -> Result<Response<Self::LoadSigningKeyStream>, tonic::Status> {
        let mut in_stream = request.into_inner();
        let key_pair = self.signing_key_pair.clone();
        let session_config = self.get_session_config();
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let result = async {
                let mut session = ClientSession::create(session_config)
                    .context("failed to create ClientSession")?;

                // Get the initial message. Unlike all other steps in the
                // protocol, the ClientSession will return an error if
                // `get_outgoing_message()` is called an extra time before a
                // response is received from the server.
                let request = session
                    .get_outgoing_message()?
                    .context("failed to get first message from ClientSession")
                    .context(Code::InvalidArgument)?;
                tx.send(Ok(request.convert()?)).await?;
                session = initialize_session(session, &mut in_stream, &tx).await?;

                // Process the final message.
                let response =
                    in_stream.message().await?.context("server unexpectedly closed stream")?;
                session.put_incoming_message(response.convert()?)?;
                let response = session.read()?.context("ClientSession::read returned None")?;
                *key_pair.lock().unwrap() = Some(Arc::new(
                    ciborium::from_reader(response.plaintext.as_slice())
                        .context("failed to decode KeyPair")?,
                ));
                Ok::<_, anyhow::Error>(())
            };
            if let Err(err) = result.await {
                let _ = tx.send(Err(Self::convert_error(err))).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_status(
        &self,
        _request: Request<GetStatusRequest>,
    ) -> Result<Response<GetStatusResponse>, tonic::Status> {
        Ok(Response::new(GetStatusResponse {
            version_fprint: self.version_fprint.clone(),
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
