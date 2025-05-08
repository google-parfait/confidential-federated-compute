// Copyright 2025 Google LLC.
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

use std::{pin::Pin, sync::Arc};

use googletest::prelude::*;
use matchers::{code, has_context};
use mockall::{predicate::eq as request_eq, PredicateBooleanExt};
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_proto_rust::oak::{attestation::v1::ReferenceValues, session::v1::PlaintextMessage};
use oak_session::{ProtocolEngine, ServerSession, Session};
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use session_config::create_session_config;
use session_test_utils::{
    get_test_attester, get_test_endorser, get_test_reference_values, get_test_signer, FakeClock,
    TestSigner,
};
use session_v1_service_proto::{
    oak::services::{
        oak_session_v1_service_client::OakSessionV1ServiceClient,
        oak_session_v1_service_server::{OakSessionV1Service, OakSessionV1ServiceServer},
    },
    session_proto::oak::session::v1::{SessionRequest, SessionResponse},
};
use storage_client::{GrpcStorageClient, StorageClient};
use storage_proto::{
    confidential_federated_compute::kms::{
        read_request, storage_request, storage_response, update_request, ReadRequest, ReadResponse,
        StorageRequest, StorageResponse, UpdateRequest, UpdateResponse,
    },
    status_proto::google::rpc::Status,
    timestamp_proto::google::protobuf::Timestamp,
};
use tokio::{
    net::TcpListener,
    sync::mpsc,
    time::{timeout, Duration},
};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::{server::TcpIncoming, Server};
use tracing::debug;

#[mockall::automock]
trait Storage {
    fn call(
        &self,
        request: storage_request::Kind,
    ) -> std::result::Result<storage_response::Kind, tonic::Status>;
}

/// A fake OakSessionV1Service implementation that delegates calls to a
/// MockStorage implementation. This allows tests to mock gRPC client/server
/// interactions over an encrypted Oak session without too much complexity.
struct FakeServer {
    storage: Arc<MockStorage>,
    attester: Arc<dyn Attester>,
    endorser: Arc<dyn Endorser>,
    signer: TestSigner,
    reference_values: ReferenceValues,
    clock: Arc<FakeClock>,
}

impl FakeServer {
    fn new(storage: MockStorage) -> Self {
        Self {
            storage: Arc::new(storage),
            attester: get_test_attester(),
            endorser: get_test_endorser(),
            signer: get_test_signer(),
            reference_values: get_test_reference_values().convert().unwrap(),
            clock: Arc::new(FakeClock { milliseconds_since_epoch: 0 }),
        }
    }
}

#[tonic::async_trait]
impl OakSessionV1Service for FakeServer {
    type StreamStream =
        Pin<Box<dyn Stream<Item = std::result::Result<SessionResponse, tonic::Status>> + Send>>;

    async fn stream(
        &self,
        request: tonic::Request<tonic::Streaming<SessionRequest>>,
    ) -> std::result::Result<tonic::Response<Self::StreamStream>, tonic::Status> {
        let mut session = create_session_config(
            &self.attester,
            &self.endorser,
            Box::new(self.signer.clone()),
            self.reference_values.clone(),
            self.clock.clone(),
        )
        .and_then(ServerSession::create)
        .expect("failed to create ServerSession");

        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(128);
        let storage = self.storage.clone();
        tokio::spawn(async move {
            while let Some(msg) = in_stream.next().await {
                debug!("Server received SessionRequest: {:?}", msg);
                session
                    .put_incoming_message(
                        oak_proto_rust::oak::session::v1::SessionRequest::decode(
                            msg.unwrap().encode_to_vec().as_slice(),
                        )
                        .unwrap(),
                    )
                    .expect("failed to put incoming message");
                if session.is_open() {
                    if let Some(msg) = session.read().expect("failed to read from session") {
                        let request = StorageRequest::decode(msg.plaintext.as_slice()).unwrap();
                        debug!("Decoded StorageRequest: {:?}", request);
                        match storage.call(request.kind.expect("no request kind")) {
                            Ok(response) => {
                                let response = StorageResponse {
                                    correlation_id: request.correlation_id,
                                    kind: Some(response),
                                };
                                debug!("Returning StorageResponse: {:?}", response);
                                session
                                    .write(PlaintextMessage { plaintext: response.encode_to_vec() })
                                    .expect("failed to write to session");
                            }
                            Err(err) => {
                                debug!("Returning error: {:?}", err);
                                tx.send(Err(err)).await.expect("failed to send message");
                                return;
                            }
                        };
                    }
                }

                while let Some(response) = session.get_outgoing_message().unwrap() {
                    let response =
                        SessionResponse::decode(response.encode_to_vec().as_slice()).unwrap();
                    debug!("Server sending SessionResponse: {:?}", response);
                    tx.send(Ok(response)).await.expect("failed to send message");
                }
            }
        });
        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A Signer that calls `block_on` to emulate
/// `oak_sdk_containers::InstanceSigner`.
#[derive(Clone)]
struct BlockingSigner {
    inner: TestSigner,
}
impl oak_crypto::signer::Signer for BlockingSigner {
    fn sign(&self, message: &[u8]) -> Vec<u8> {
        tokio::runtime::Handle::current().block_on(async { self.inner.sign(message) })
    }
}

/// Starts a fake server that delegates to the given MockStorage object. Returns
/// a StorageClient connected to the server and a handle that will stop the
/// server when dropped.
async fn start_server<F: Fn() -> UpdateRequest + Send + 'static>(
    storage: MockStorage,
    init_fn: F,
) -> (GrpcStorageClient, AbortOnDropHandle<()>) {
    let listener = TcpListener::bind("[::]:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = FakeServer::new(storage);
    // Test compatibility with `oak_sdk_containers::InstanceSigner`.
    let signer = BlockingSigner { inner: server.signer.clone() };
    let client = GrpcStorageClient::new(
        OakSessionV1ServiceClient::connect(format!("http://{addr}")).await.unwrap(),
        init_fn,
        server.attester.clone(),
        server.endorser.clone(),
        signer,
        server.reference_values.clone(),
        server.clock.clone(),
    );
    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(OakSessionV1ServiceServer::new(server))
            .serve_with_incoming(TcpIncoming::from_listener(listener, true, None).unwrap())
            .await
            .unwrap();
    });
    (client, AbortOnDropHandle::new(handle))
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn init_succeeds() {
    let mut storage = MockStorage::new();
    let init_fn = || UpdateRequest {
        updates: vec![update_request::Update { key: b"key".into(), ..Default::default() }],
    };
    // The initialization request should be sent once before any other requests.
    let mut seq = mockall::Sequence::new();
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(init_fn())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Read(ReadRequest::default())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Read(ReadResponse::default())));

    let (client, _server_handle) = start_server(storage, init_fn).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.read(ReadRequest::default())).await,
        ok(ok(anything()))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn init_ignores_failed_precondition() {
    let mut storage = MockStorage::new();
    // A FAILED_PRECONDITION error should be an accepted initialization result
    // since it indicates that initialization was already performed.
    storage.expect_call().times(1).return_const(Ok(storage_response::Kind::Error(Status {
        code: tonic::Code::FailedPrecondition.into(),
        ..Default::default()
    })));
    storage.expect_call().return_const(Ok(storage_response::Kind::Read(ReadResponse::default())));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.read(ReadRequest::default())).await,
        ok(ok(anything()))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn init_retries_other_errors() {
    let mut storage = MockStorage::new();
    let init_fn = || UpdateRequest {
        updates: vec![update_request::Update { key: b"key".into(), ..Default::default() }],
    };
    let mut seq = mockall::Sequence::new();
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(init_fn())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Error(Status {
            code: tonic::Code::Internal.into(),
            ..Default::default()
        })));
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(init_fn())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    // Once the init request succeeds, the read request should be sent.
    storage
        .expect_call()
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Read(ReadResponse::default())));

    let (client, _server_handle) = start_server(storage, init_fn).await;
    expect_that!(
        timeout(Duration::from_secs(2), client.read(ReadRequest::default())).await,
        ok(ok(anything()))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn read_succeeds() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    let read_request = ReadRequest {
        ranges: vec![read_request::Range { start: b"key".into(), ..Default::default() }],
    };
    let read_response = ReadResponse {
        now: Some(Timestamp { seconds: 1000, ..Default::default() }),
        ..Default::default()
    };
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Read(read_request.clone())))
        .return_const(Ok(storage_response::Kind::Read(read_response.clone())));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(2), client.read(read_request)).await,
        ok(ok(eq(read_response)))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn read_with_error_fails() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    storage.expect_call().return_const(Ok(storage_response::Kind::Error(Status {
        code: tonic::Code::Internal as i32,
        message: "message".into(),
        ..Default::default()
    })));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.read(ReadRequest::default())).await,
        ok(err(all!(code(tonic::Code::Internal), has_context(eq("message")))))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn read_with_wrong_response_kind_fails() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    // Return an UpdateResponse instead of a ReadResponse.
    storage
        .expect_call()
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.read(ReadRequest::default())).await,
        ok(err(has_context(contains_substring("unexpected StorageResponse.kind"))))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn update_succeeds() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    let update_request = UpdateRequest {
        updates: vec![update_request::Update { key: b"key".into(), ..Default::default() }],
    };
    let update_response = UpdateResponse::default();
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(update_request.clone())))
        .return_const(Ok(storage_response::Kind::Update(update_response.clone())));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.update(update_request)).await,
        ok(ok(eq(update_response)))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn update_with_error_fails() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    storage.expect_call().return_const(Ok(storage_response::Kind::Error(Status {
        code: tonic::Code::Internal as i32,
        message: "message".into(),
        ..Default::default()
    })));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.update(UpdateRequest::default())).await,
        ok(err(all!(code(tonic::Code::Internal), has_context(eq("message")))))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn update_with_wrong_response_kind_fails() {
    let mut storage = MockStorage::new();
    storage
        .expect_call()
        .times(1)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    // Return a ReadResponse instead of an UpdateResponse.
    storage.expect_call().return_const(Ok(storage_response::Kind::Read(ReadResponse::default())));

    let (client, _server_handle) = start_server(storage, UpdateRequest::default).await;
    expect_that!(
        timeout(Duration::from_secs(1), client.update(UpdateRequest::default())).await,
        ok(err(has_context(contains_substring("unexpected StorageResponse.kind"))))
    );
}

#[googletest::test]
#[test_log::test(tokio::test)]
async fn retry_on_connection_failure() {
    let mut storage = MockStorage::new();
    let init_fn = || UpdateRequest {
        updates: vec![update_request::Update { key: b"init".into(), ..Default::default() }],
    };
    let mut seq = mockall::Sequence::new();
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(init_fn())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    let read_request = ReadRequest {
        ranges: vec![read_request::Range { start: b"read".into(), ..Default::default() }],
    };
    let update_request = UpdateRequest {
        updates: vec![update_request::Update { key: b"update".into(), ..Default::default() }],
    };
    storage
        .expect_call()
        // The read and update requests can be sent in any order; fail whichever
        // comes first.
        .with(
            request_eq(storage_request::Kind::Read(read_request.clone()))
                .or(request_eq(storage_request::Kind::Update(update_request.clone()))),
        )
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Err(tonic::Status::internal("message")));

    // After the error, a new connection should be established and the requests
    // should be retried.
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(init_fn())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Read(read_request.clone())))
        .times(1)
        .in_sequence(&mut seq)
        .return_const(Ok(storage_response::Kind::Read(ReadResponse::default())));
    storage
        .expect_call()
        .with(request_eq(storage_request::Kind::Update(update_request.clone())))
        .times(1)
        // AFAIK, mockall cannot represent that this call and the previous one
        // can happen in any order, so long as they're after the second
        // initialization request. We therefore require ordering on only the
        // read call and assume that the update call behaves similarly.
        .return_const(Ok(storage_response::Kind::Update(UpdateResponse::default())));

    let (client, _server_handle) = start_server(storage, init_fn).await;
    let (read_response, update_response) = tokio::join!(
        timeout(Duration::from_secs(2), client.read(read_request)),
        timeout(Duration::from_secs(2), client.update(update_request)),
    );
    expect_that!(read_response, ok(ok(eq(ReadResponse::default()))));
    expect_that!(update_response, ok(ok(eq(UpdateResponse::default()))));
}
