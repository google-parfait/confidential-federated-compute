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

use std::sync::Arc;

use anyhow::{bail, Context};
use attestation_transparency_service::{
    build_signed_payload_sig_structure, AttestationTransparencyService,
};
use attestation_transparency_service_proto::{
    fcp::confidentialcompute::{
        attestation_transparency_service_client::AttestationTransparencyServiceClient,
        attestation_transparency_service_server::AttestationTransparencyServiceServer,
        create_signing_key_request, create_signing_key_request::CommitKey,
        create_signing_key_response, CreateSigningKeyRequest, GetStatusRequest, GetStatusResponse,
    },
    payload_transparency_proto::{
        fcp::confidentialcompute::{
            signed_payload::{signature, signature::Headers, Signature},
            SignedPayload,
        },
        key_proto::fcp::confidentialcompute::{key, Key},
    },
};
use bssl_crypto::{ec, ecdsa};
use googletest::prelude::*;
use matchers::{has_context, when_deserialized};
use oak_crypto::signer::Signer;
use oak_proto_rust::oak::attestation::v1::Stage0Measurements;
use oak_sdk_standalone::{Standalone, StandaloneBuilder};
use oak_session::session_binding::{SessionBinder, SignatureBinder};
use oak_time::{clock::FixedClock, UNIX_EPOCH};
use p256::ecdsa::{SigningKey, VerifyingKey};
use payload_signer::PayloadSigner;
use prost::Message;
use rand_core::OsRng;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::task::AbortOnDropHandle;
use tonic::{
    transport::{server::TcpIncoming, Server},
    Response,
};

struct TestSigner {
    signing_key: SigningKey,
}
#[async_trait::async_trait]
impl oak_sdk_containers::Signer for TestSigner {
    async fn sign(
        &self,
        data: &[u8],
    ) -> anyhow::Result<oak_proto_rust::oak::crypto::v1::Signature> {
        Ok(oak_proto_rust::oak::crypto::v1::Signature { signature: self.signing_key.sign(data) })
    }
}

/// A SessionBinder that calls `block_on` to emulate
/// `oak_sdk_containers::InstanceSessionBinder`.
#[derive(Clone)]
struct BlockingSessionBinder {
    inner: Arc<SignatureBinder>,
}
impl SessionBinder for BlockingSessionBinder {
    fn bind(&self, bound_data: &[u8]) -> Vec<u8> {
        tokio::runtime::Handle::current().block_on(async { self.inner.bind(bound_data) })
    }
}

async fn start_server() -> (
    AttestationTransparencyServiceClient<tonic::transport::Channel>,
    impl PayloadSigner,
    AbortOnDropHandle<()>,
) {
    start_server_with_standalone(
        Standalone::builder()
            .stage0_measurements(Stage0Measurements {
                setup_data_digest: b"setup-data-digest".into(),
                kernel_measurement: b"kernel-measurement".into(),
                ram_disk_digest: b"ram-disk-digest".into(),
                memory_map_digest: b"memory-map-digest".into(),
                acpi_digest: b"acpi-digest".into(),
                ..Default::default()
            })
            .stage1_system_image(b"stage1-system-image")
            .application_image(b"application-image")
            .application_config(b"application-config".into()),
    )
    .await
}

async fn start_server_with_standalone(
    standalone: StandaloneBuilder<'_>,
) -> (
    AttestationTransparencyServiceClient<tonic::transport::Channel>,
    impl PayloadSigner,
    AbortOnDropHandle<()>,
) {
    // Add known signing and session binding keys..
    let signing_key = SigningKey::random(&mut OsRng);
    let session_binding_key = SigningKey::random(&mut OsRng);
    let standalone = standalone
        .signing_key_pair(Some((signing_key.clone(), VerifyingKey::from(&signing_key))))
        .session_binding_key_pair(Some((
            session_binding_key.clone(),
            VerifyingKey::from(&session_binding_key),
        )))
        .build()
        .unwrap();

    let ats = AttestationTransparencyService::create(
        TestSigner { signing_key },
        &standalone.endorsed_evidence(),
        Arc::new(BlockingSessionBinder {
            inner: Arc::new(SignatureBinder::new(Box::new(session_binding_key))),
        }),
        Arc::new(FixedClock::at_instant(UNIX_EPOCH)),
    )
    .unwrap();
    let signer = ats.signer();

    let listener = TcpListener::bind("[::]:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client =
        AttestationTransparencyServiceClient::connect(format!("http://{addr}")).await.unwrap();
    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(AttestationTransparencyServiceServer::new(ats))
            .serve_with_incoming(TcpIncoming::from_listener(listener, true, None).unwrap())
            .await
            .unwrap();
    });
    (client, signer, AbortOnDropHandle::new(handle))
}

async fn create_signing_key(
    client: &mut AttestationTransparencyServiceClient<tonic::transport::Channel>,
) -> anyhow::Result<SignedPayload> {
    let (server_tx, rx) = mpsc::channel(1);
    let mut responses = client
        .create_signing_key(ReceiverStream::new(rx))
        .await
        .context("CreateSigningKey failed")?
        .into_inner();

    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CreateKey(Default::default())),
        })
        .await
        .context("failed to send CreateKey")?;

    let response =
        responses.message().await.context("CreateKey failed")?.context("stream closed")?;
    let unpublished_key = match response.kind {
        Some(create_signing_key_response::Kind::UnpublishedKey(k)) => k,
        _ => bail!("unexpected CreateSigningKeyResponse: {:?}", response),
    };

    let verifying_key = SignedPayload {
        payload: unpublished_key.verifying_key,
        signatures: vec![Signature { headers: b"headers".into(), ..Default::default() }],
    };
    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CommitKey(CommitKey {
                verifying_key: Some(verifying_key.clone()),
            })),
        })
        .await
        .context("failed to send CommitKey")?;
    if let Some(msg) = responses.message().await.context("CommitKey failed")? {
        bail!("stream not closed; received {:?}", msg);
    }
    Ok(verifying_key)
}

#[googletest::test]
#[tokio::test]
async fn sign_fails_without_key() {
    let (_, signer, _server_handle) = start_server().await;
    expect_that!(
        signer.sign(b"headers", b"payload"),
        err(has_context(contains_substring("AttestationTransparencyService not initialized")))
    );
}

#[googletest::test]
#[tokio::test]
async fn sign_succeeds_with_key() {
    let (mut client, signer, _server_handle) = start_server().await;

    let headers = Headers { claims: vec!["claim1".into(), "claim2".into()], ..Default::default() };
    let verifying_key = create_signing_key(&mut client).await.unwrap();
    let ats_key = Key::decode(verifying_key.payload.as_slice()).unwrap();

    let signature = signer.sign(&headers.encode_to_vec(), b"payload").expect("signing failed");
    assert_that!(
        signature,
        matches_pattern!(Signature {
            headers: when_deserialized::<Headers>(eq(Headers {
                algorithm: ats_key.algorithm,
                ..headers
            })),
            signature: some(matches_pattern!(signature::Signature::RawSignature(not(empty())))),
            verifier: some(eq(signature::Verifier::VerifyingKey(verifying_key))),
        })
    );

    // The result should be signed by the ATS key.
    let raw_signature = match &signature.signature {
        Some(signature::Signature::RawSignature(raw)) => raw,
        _ => unreachable!(),
    };
    assert_that!(ats_key.algorithm, eq(key::Algorithm::EcdsaP256 as i32));
    let public_key =
        ecdsa::PublicKey::<ec::P256>::from_x962_uncompressed(&ats_key.key_material).unwrap();
    expect_that!(
        public_key.verify_p1363(
            &build_signed_payload_sig_structure(&signature.headers, b"payload"),
            raw_signature
        ),
        ok(anything())
    );
}

#[googletest::test]
#[tokio::test]
async fn create_signing_key_succeeds() {
    let (mut client, _, _server_handle) = start_server().await;

    let verifying_key = create_signing_key(&mut client).await.unwrap();
    expect_that!(
        verifying_key.payload,
        when_deserialized(matches_pattern!(Key {
            algorithm: eq(key::Algorithm::EcdsaP256 as i32),
            purpose: some(eq(key::Purpose::Verify as i32)),
            key_id: not(empty()),
            key_material: not(empty()),
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn create_signing_key_fails_with_wrong_first_message() {
    let (mut client, _, _server_handle) = start_server().await;

    let (server_tx, rx) = mpsc::channel(1);
    let mut responses =
        client.create_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();

    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CommitKey(Default::default())),
        })
        .await
        .unwrap();
    expect_that!(
        responses.message().await,
        err(displays_as(contains_substring(
            "first CreateSigningKeyRequest must contain CreateKey"
        )))
    );

    // A key should not have been created.
    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: none() }))
    );
}

#[googletest::test]
#[tokio::test]
async fn create_signing_key_fails_with_modified_key() {
    let (mut client, _, _server_handle) = start_server().await;

    let (server_tx, rx) = mpsc::channel(1);
    let mut responses =
        client.create_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();

    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CreateKey(Default::default())),
        })
        .await
        .unwrap();
    let unpublished_key = match responses.message().await.unwrap().unwrap().kind {
        Some(create_signing_key_response::Kind::UnpublishedKey(k)) => k,
        _ => unreachable!(),
    };

    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CommitKey(CommitKey {
                verifying_key: Some(SignedPayload {
                    // Modify the payload.
                    payload: [unpublished_key.verifying_key.as_slice(), b"x"].concat(),
                    signatures: vec![Signature {
                        headers: b"headers".into(),
                        ..Default::default()
                    }],
                }),
            })),
        })
        .await
        .unwrap();
    expect_that!(
        responses.message().await,
        err(displays_as(contains_substring("commit_key.verifying_key payload does not match")))
    );

    // A key should not have been created.
    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: none() }))
    );
}

#[googletest::test]
#[tokio::test]
async fn create_signing_key_aborted() {
    let (mut client, _, _server_handle) = start_server().await;

    let (server_tx, rx) = mpsc::channel(1);
    let mut responses =
        client.create_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();

    server_tx
        .send(CreateSigningKeyRequest {
            kind: Some(create_signing_key_request::Kind::CreateKey(Default::default())),
        })
        .await
        .unwrap();
    assert_that!(responses.message().await, ok(some(anything())));

    // Abort the stream without sending a CommitKey message.
    drop(server_tx);
    expect_that!(
        responses.message().await,
        err(displays_as(contains_substring("CreateSigningKey aborted")))
    );

    // A key should not have been created.
    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: none() }))
    );
}

#[googletest::test]
#[tokio::test]
async fn create_signing_key_replaces_existing_key() {
    let (mut client, _, _server_handle) = start_server().await;

    let verifying_key1 = create_signing_key(&mut client).await.unwrap();
    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: some(eq(verifying_key1.clone())) }))
    );

    let verifying_key2 = create_signing_key(&mut client).await.unwrap();
    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: some(eq(verifying_key2.clone())) }))
    );
    expect_that!(verifying_key1, not(eq(verifying_key2)));
}

#[googletest::test]
#[tokio::test]
async fn share_and_load_signing_key_succeeds() {
    let (mut client1, _, _server1_handle) = start_server().await;
    let verifying_key = create_signing_key(&mut client1).await.unwrap();
    let ats_key = Key::decode(verifying_key.payload.as_slice()).unwrap();

    // Share the signing key with a second server.
    let (mut client2, signer, _server2_handle) = start_server().await;
    let (share_tx, rx) = mpsc::channel(1);
    let mut share_responses =
        client1.share_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let (load_tx, rx) = mpsc::channel(1);
    let mut load_responses =
        client2.load_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let forward_load_messages = async move {
        while let Some(msg) = load_responses.message().await? {
            share_tx.send(msg).await?;
        }
        Ok::<_, anyhow::Error>(())
    };
    let forward_share_messages = async move {
        while let Some(msg) = share_responses.message().await? {
            load_tx.send(msg).await?;
        }
        Ok::<_, anyhow::Error>(())
    };
    tokio::try_join!(forward_load_messages, forward_share_messages).expect("failed to share key");

    // The second server should now have the key.
    expect_that!(
        client2.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: some(eq(verifying_key.clone())) }))
    );

    // The second server's signer should now work.
    let headers = Headers { claims: vec!["claim1".into(), "claim2".into()], ..Default::default() };
    let signature = signer.sign(&headers.encode_to_vec(), b"payload").expect("signing failed");
    expect_that!(
        signature,
        matches_pattern!(Signature {
            headers: when_deserialized::<Headers>(eq(Headers {
                algorithm: ats_key.algorithm,
                ..headers
            })),
            signature: some(matches_pattern!(signature::Signature::RawSignature(not(empty())))),
            verifier: some(eq(signature::Verifier::VerifyingKey(verifying_key))),
        })
    );

    // The result should be signed by the ATS key.
    let raw_signature = match &signature.signature {
        Some(signature::Signature::RawSignature(raw)) => raw,
        _ => unreachable!(),
    };
    assert_that!(ats_key.algorithm, eq(key::Algorithm::EcdsaP256 as i32));
    let public_key =
        ecdsa::PublicKey::<ec::P256>::from_x962_uncompressed(&ats_key.key_material).unwrap();
    expect_that!(
        public_key.verify_p1363(
            &build_signed_payload_sig_structure(&signature.headers, b"payload"),
            raw_signature
        ),
        ok(anything())
    );
}

#[googletest::test]
#[tokio::test]
async fn share_signing_key_fails_without_key() {
    let (mut client, _, _server_handle) = start_server().await;

    let (_, rx) = mpsc::channel(1);
    expect_that!(
        client.share_signing_key(ReceiverStream::new(rx)).await,
        err(displays_as(contains_substring("no signing key has been created")))
    );
}

#[googletest::test]
#[tokio::test]
async fn share_and_load_signing_key_fails_with_different_evidence() {
    let standalone1 = Standalone::builder()
        .stage0_measurements(Stage0Measurements {
            setup_data_digest: b"setup-data-digest".into(),
            kernel_measurement: b"kernel-measurement".into(),
            ram_disk_digest: b"ram-disk-digest".into(),
            memory_map_digest: b"memory-map-digest".into(),
            acpi_digest: b"acpi-digest".into(),
            ..Default::default()
        })
        .stage1_system_image(b"stage1-system-image")
        .application_image(b"application-image1")
        .application_config(b"application-config".into());
    let (mut client1, _, _server1_handle) = start_server_with_standalone(standalone1).await;

    let standalone2 = Standalone::builder()
        .stage0_measurements(Stage0Measurements {
            setup_data_digest: b"setup-data-digest".into(),
            kernel_measurement: b"kernel-measurement".into(),
            ram_disk_digest: b"ram-disk-digest".into(),
            memory_map_digest: b"memory-map-digest".into(),
            acpi_digest: b"acpi-digest".into(),
            ..Default::default()
        })
        .stage1_system_image(b"stage1-system-image")
        .application_image(b"application-image2")
        .application_config(b"application-config".into());
    let (mut client2, _, _server2_handle) = start_server_with_standalone(standalone2).await;

    // Attempt to share the signing key with the second server.
    create_signing_key(&mut client1).await.unwrap();
    let (share_tx, rx) = mpsc::channel(1);
    let mut share_responses =
        client1.share_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let (load_tx, rx) = mpsc::channel(1);
    let mut load_responses =
        client2.load_signing_key(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let forward_load_messages = async move {
        while let Some(msg) = load_responses.message().await? {
            share_tx.send(msg).await?;
        }
        Ok::<_, anyhow::Error>(())
    };
    let forward_share_messages = async move {
        while let Some(msg) = share_responses.message().await? {
            load_tx.send(msg).await?;
        }
        Ok::<_, anyhow::Error>(())
    };
    expect_that!(
        tokio::try_join!(forward_load_messages, forward_share_messages),
        err(displays_as(contains_substring("container layer")))
    );

    // The second server should not have loaded a key.
    expect_that!(
        client2.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: none() }))
    );
}

#[googletest::test]
#[tokio::test]
async fn get_status_without_key() {
    let (mut client, _, _server_handle) = start_server().await;

    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse {
            verifying_key: none(),
            version_fprint: not(empty())
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn get_status_with_key() {
    let (mut client, _, _server_handle) = start_server().await;
    let verifying_key = create_signing_key(&mut client).await.unwrap();

    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse {
            verifying_key: some(eq(verifying_key)),
            version_fprint: not(empty()),
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn version_fprint_depends_on_endorsed_evidence() {
    let (mut client1, _, _server_handle1) = start_server_with_standalone(
        Standalone::builder().application_image(b"application-image1"),
    )
    .await;
    let fprint =
        client1.get_status(GetStatusRequest::default()).await.unwrap().into_inner().version_fprint;

    // The fprint should be consistent for the same server.
    expect_that!(
        client1.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { version_fprint: eq(fprint.clone()) }))
    );

    // The fprint should match a different server with the same evidence.
    let (mut client2, _, _server_handle2) = start_server_with_standalone(
        Standalone::builder().application_image(b"application-image1"),
    )
    .await;
    expect_that!(
        client2.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { version_fprint: eq(fprint.clone()) }))
    );

    // The fprint should not match a server with different evidence.
    let (mut client3, _, _server_handle3) = start_server_with_standalone(
        Standalone::builder().application_image(b"application-image2"),
    )
    .await;
    expect_that!(
        client3.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { version_fprint: not(eq(fprint)) }))
    );
}

#[googletest::test]
fn build_signed_payload_sig_structure_succeeds() {
    expect_that!(
        build_signed_payload_sig_structure(b"hdrs", b"payload"),
        eq(b"\x0dSignedPayload\x04hdrs\x07payload")
    );
}
