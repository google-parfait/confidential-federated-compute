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
use oak_proto_rust::oak::session::v1::EndorsedEvidence;
use payload_signer::PayloadSigner;
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use session_test_utils::{get_test_endorsements, get_test_evidence, get_test_signer};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::task::AbortOnDropHandle;
use tonic::{
    transport::{server::TcpIncoming, Server},
    Code, Response, Status,
};

struct TestSigner {}
#[async_trait::async_trait]
impl oak_sdk_containers::Signer for TestSigner {
    async fn sign(
        &self,
        data: &[u8],
    ) -> anyhow::Result<oak_proto_rust::oak::crypto::v1::Signature> {
        Ok(oak_proto_rust::oak::crypto::v1::Signature { signature: get_test_signer().sign(data) })
    }
}

async fn start_server() -> (
    AttestationTransparencyServiceClient<tonic::transport::Channel>,
    impl PayloadSigner,
    AbortOnDropHandle<()>,
) {
    let ats = AttestationTransparencyService::create(
        TestSigner {},
        &EndorsedEvidence {
            evidence: Some(get_test_evidence().convert().unwrap()),
            endorsements: Some(get_test_endorsements().convert().unwrap()),
        },
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
async fn share_signing_key_unimplemented() {
    let (mut client, _, _server_handle) = start_server().await;

    let (_, rx) = mpsc::channel(1);
    expect_that!(
        client.share_signing_key(ReceiverStream::new(rx)).await,
        err(all!(
            property!(Status.code(), eq(Code::Unimplemented)),
            displays_as(contains_substring("ShareSigningKey is unimplemented")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn load_signing_key_unimplemented() {
    let (mut client, _, _server_handle) = start_server().await;

    let (_, rx) = mpsc::channel(1);
    expect_that!(
        client.load_signing_key(ReceiverStream::new(rx)).await,
        err(all!(
            property!(Status.code(), eq(Code::Unimplemented)),
            displays_as(contains_substring("LoadSigningKey is unimplemented")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn get_status_without_key() {
    let (mut client, _, _server_handle) = start_server().await;

    expect_that!(
        client.get_status(GetStatusRequest::default()).await.map(Response::into_inner),
        ok(matches_pattern!(GetStatusResponse { verifying_key: none(), version_fprint: empty() }))
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
            version_fprint: empty(),
        }))
    );
}

#[googletest::test]
fn build_signed_payload_sig_structure_succeeds() {
    expect_that!(
        build_signed_payload_sig_structure(b"hdrs", b"payload"),
        eq(b"\x0dSignedPayload\x04hdrs\x07payload")
    );
}
