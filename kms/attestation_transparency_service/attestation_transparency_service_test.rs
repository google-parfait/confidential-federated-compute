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

use attestation_transparency_service::AttestationTransparencyService;
use attestation_transparency_service_proto::fcp::confidentialcompute::{
    attestation_transparency_service_client::AttestationTransparencyServiceClient,
    attestation_transparency_service_server::AttestationTransparencyServiceServer,
    GetStatusRequest,
};
use googletest::prelude::*;
use matchers::has_context;
use payload_signer::PayloadSigner;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::task::AbortOnDropHandle;
use tonic::{
    transport::{server::TcpIncoming, Server},
    Code, Status,
};

async fn start_server() -> (
    AttestationTransparencyServiceClient<tonic::transport::Channel>,
    impl PayloadSigner,
    AbortOnDropHandle<()>,
) {
    let ats = AttestationTransparencyService::default();
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
async fn create_signing_key_unimplemented() {
    let (mut client, _, _server_handle) = start_server().await;

    let (_, rx) = mpsc::channel(1);
    expect_that!(
        client.create_signing_key(ReceiverStream::new(rx)).await,
        err(all!(
            property!(Status.code(), eq(Code::Unimplemented)),
            displays_as(contains_substring("CreateSigningKey is unimplemented")),
        ))
    );
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
async fn get_status_unimplemented() {
    let (mut client, _, _server_handle) = start_server().await;

    expect_that!(
        client.get_status(GetStatusRequest::default()).await,
        err(all!(
            property!(Status.code(), eq(Code::Unimplemented)),
            displays_as(contains_substring("GetStatus is unimplemented")),
        ))
    );
}
