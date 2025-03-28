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

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use hashbrown::HashMap;
use log::{error, info, warn};
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification_types::util::Clock;
use oak_crypto::signer::Signer;
use oak_proto_rust::oak::{
    attestation::v1::ReferenceValues,
    session::v1::{PlaintextMessage, SessionResponse},
};
use oak_session::{ClientSession, ProtocolEngine, Session};
use prost::Message;
use session_config::create_session_config;
use session_v1_service_proto::{
    oak::services::oak_session_v1_service_client::OakSessionV1ServiceClient,
    session_proto::oak::session::v1::SessionRequest,
};
use storage_proto::confidential_federated_compute::kms::{
    storage_request, storage_response, ReadRequest, ReadResponse, StorageRequest, StorageResponse,
    UpdateRequest, UpdateResponse,
};
use tokio::{
    select,
    sync::{mpsc, oneshot},
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::task::AbortOnDropHandle;
use tonic::Code;

// A client for sending requests to the StorageActor over a secure channel. This
// struct handles multiplexing requests over a single stream and retrying
// requests if there are connection or protocol errors, including errors due to
// the TCP leader changing.
pub trait StorageClient {
    /// Performs a read operation.
    fn read(
        &self,
        request: ReadRequest,
    ) -> impl std::future::Future<Output = Result<ReadResponse>> + Send;

    /// Performs an update operation.
    fn update(
        &self,
        request: UpdateRequest,
    ) -> impl std::future::Future<Output = Result<UpdateResponse>> + Send;
}

// A StorageClient that communicates with the TCP leader via gRPC.
pub struct GrpcStorageClient {
    sender: mpsc::UnboundedSender<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    _handle: AbortOnDropHandle<()>,
}

impl GrpcStorageClient {
    pub fn new<S: Signer + Clone + 'static>(
        client: OakSessionV1ServiceClient<tonic::transport::Channel>,
        init_fn: impl Fn() -> UpdateRequest + Send + 'static,
        attester: Arc<dyn Attester>,
        endorser: Arc<dyn Endorser>,
        signer: S,
        reference_values: ReferenceValues,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let sender = tx.clone();
        let handle = tokio::spawn(async move {
            MessageSender {
                tx,
                rx,
                client,
                init_fn,
                attester,
                endorser,
                signer,
                reference_values,
                clock,
                pending_requests: HashMap::new(),
                next_correlation_id: 1,
            }
            .run()
            .await
        });
        Self { sender, _handle: AbortOnDropHandle::new(handle) }
    }

    async fn send_request(&self, request: StorageRequest) -> Result<StorageResponse> {
        let (tx, rx) = oneshot::channel();
        self.sender.send((request, tx)).context("failed to send request")?;
        let response = rx.await.context("failed to receive response")??;
        match response.kind {
            Some(storage_response::Kind::Error(err)) => {
                Err(anyhow::Error::msg(err.message)).context(Code::from(err.code))
            }
            _ => Ok(response),
        }
    }
}

impl StorageClient for GrpcStorageClient {
    async fn read(&self, request: ReadRequest) -> Result<ReadResponse> {
        let response = self
            .send_request(StorageRequest {
                kind: Some(storage_request::Kind::Read(request)),
                ..Default::default()
            })
            .await?;
        match response.kind {
            Some(storage_response::Kind::Read(response)) => Ok(response),
            _ => Err(anyhow!("unexpected StorageResponse.kind during read")),
        }
    }

    async fn update(&self, request: UpdateRequest) -> Result<UpdateResponse> {
        let response = self
            .send_request(StorageRequest {
                kind: Some(storage_request::Kind::Update(request)),
                ..Default::default()
            })
            .await?;
        match response.kind {
            Some(storage_response::Kind::Update(response)) => Ok(response),
            _ => Err(anyhow!("unexpected StorageResponse.kind during update")),
        }
    }
}

struct MessageSender<InitFn, S> {
    tx: mpsc::UnboundedSender<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    rx: mpsc::UnboundedReceiver<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    client: OakSessionV1ServiceClient<tonic::transport::Channel>,
    init_fn: InitFn,
    pending_requests: HashMap<u64, (StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    next_correlation_id: u64,
    attester: Arc<dyn Attester>,
    endorser: Arc<dyn Endorser>,
    signer: S,
    reference_values: ReferenceValues,
    clock: Arc<dyn Clock>,
}

impl<InitFn, S> MessageSender<InitFn, S>
where
    InitFn: Fn() -> UpdateRequest + 'static,
    S: Signer + Clone + 'static,
{
    pub async fn run(mut self) {
        while let Err(err) = self.run_session().await {
            warn!("StreamingSession closed: {:?}", err);
            // Requeue all pending requests.
            for (id, (request, tx)) in self.pending_requests.drain() {
                if let Err(err) = self.tx.send((request, tx)) {
                    error!("failed to requeue request to {}: {:?}", id, err);
                }
            }
        }
    }

    /// Handles a single encrypted session with the server. This function will
    /// return an error if/when the server closes the stream or a protocol error
    /// occurs.
    async fn run_session(&mut self) -> Result<()> {
        let (mut session, server_tx, mut responses) = self.initialize_session().await?;
        loop {
            select! {
                // Forward requests from the (local) mpsc channel to the server.
                event = self.rx.recv() => match event {
                    Some((mut msg, tx)) => {
                        // Skip requests that have been abandoned.
                        if tx.is_closed() {
                            continue;
                        }

                        msg.correlation_id = self.next_correlation_id;
                        self.next_correlation_id = self.next_correlation_id.wrapping_add(1);
                        let encoded_msg = msg.encode_to_vec();
                        self.pending_requests.insert(msg.correlation_id, (msg, tx));
                        session.write(PlaintextMessage { plaintext: encoded_msg })?;
                        while let Some(request) = session.get_outgoing_message()? {
                            server_tx.send(Self::convert_request(&request)?).await?;
                        }
                    }
                    None => bail!("StorageClient stream unexpectedly closed"),
                },

                // Forward responses from the remote server back to the local caller.
                event = responses.message() => match event? {
                    Some(response) => {
                        session.put_incoming_message(Self::convert_response(&response)?)?;
                        let response = session.read()?
                            .ok_or_else(|| anyhow!("ClientSession::read returned None"))?;
                        let msg = StorageResponse::decode(response.plaintext.as_slice())
                            .context("failed to decode StorageResponse")?;

                        if let Some((_, tx)) = self.pending_requests.remove(&msg.correlation_id) {
                            if let Err(err) = tx.send(Ok(msg)) {
                                warn!("failed to send response to client: {:?}", err);
                            }
                        } else {
                            warn!("unexpected correlation id in response: {}", msg.correlation_id);
                        }
                    },
                    None => bail!("server unexpectedly closed stream"),
                }
            }
        }
    }

    /// Sets up and initializes a ClientSession.
    ///
    /// The session will be open when returned and the `self.init_fn`
    /// UpdateRequest will have been applied.
    async fn initialize_session(
        &mut self,
    ) -> Result<(
        ClientSession,
        mpsc::Sender<SessionRequest>,
        tonic::Streaming<
            session_v1_service_proto::session_proto::oak::session::v1::SessionResponse,
        >,
    )> {
        let mut session = create_session_config(
            &self.attester,
            &self.endorser,
            Box::new(self.signer.clone()),
            self.reference_values.clone(),
            self.clock.clone(),
        )
        .and_then(ClientSession::create)
        .context("failed to create ClientSession")?;

        // Construct the initial message. Unlike all other steps in the protocol,
        // the ClientSession will return an error if `get_outgoing_message()` is
        // called an extra time before a response is received from the server.
        let request = session
            .get_outgoing_message()?
            .ok_or_else(|| anyhow!("failed to get first message from ClientSession"))?;

        // Start the stream. https://github.com/hyperium/tonic/issues/515 means
        // that this may block until the server sends a response, so this MUST
        // happen after an initial message is added to `server_tx`.
        let (server_tx, rx) = mpsc::channel(1);
        server_tx.send(Self::convert_request(&request)?).await?;
        let mut responses = self.client.stream(ReceiverStream::new(rx)).await?.into_inner();

        // Perform the initial handshake.
        while !session.is_open() {
            let response = responses
                .message()
                .await?
                .ok_or_else(|| anyhow!("server unexpectedly closed stream"))?;
            // `oak_sdk_containers::InstanceSigner` performs blocking operations,
            // so it cannot be called from an async thread. The signer is only
            // used during the initial handshake, so it's not necessary to run
            // subsequent ClientSession interactions on a separate thread.
            let requests;
            (session, requests) = tokio::task::spawn_blocking(
                move || -> Result<(ClientSession, Vec<SessionRequest>)> {
                    session.put_incoming_message(Self::convert_response(&response)?)?;
                    let mut requests = Vec::with_capacity(1);
                    while let Some(request) = session.get_outgoing_message()? {
                        requests.push(Self::convert_request(&request)?);
                    }
                    Ok((session, requests))
                },
            )
            .await??;
            for request in requests {
                server_tx.send(request).await?;
            }
        }

        // Send the initialization request.
        session.write(PlaintextMessage {
            plaintext: StorageRequest {
                correlation_id: 0,
                kind: Some(storage_request::Kind::Update((self.init_fn)())),
            }
            .encode_to_vec(),
        })?;
        while let Some(request) = session.get_outgoing_message()? {
            server_tx.send(Self::convert_request(&request)?).await?;
        }

        // Process the initialization response. Initialization should only be
        // performed once per cluster of storage servers, so this operation will
        // frequently fail with a FAILED_PRECONDITION error.
        let response = responses
            .message()
            .await?
            .ok_or_else(|| anyhow!("server unexpectedly closed stream"))?;
        session.put_incoming_message(Self::convert_response(&response)?)?;
        let response =
            session.read()?.ok_or_else(|| anyhow!("ClientSession::read returned None"))?;
        match StorageResponse::decode(response.plaintext.as_slice()) {
            Ok(StorageResponse { correlation_id, .. }) if correlation_id != 0 => {
                bail!("unexpected correlation id in init response: {}", correlation_id);
            }
            Ok(StorageResponse { kind: Some(storage_response::Kind::Update(_)), .. }) => {
                info!("Storage initialized!");
            }
            Ok(StorageResponse { kind: Some(storage_response::Kind::Error(err)), .. }) => {
                if err.code == Code::FailedPrecondition as i32 {
                    info!("Storage already initialized");
                } else {
                    return Err(anyhow::Error::msg(err.message)
                        .context("storage initialization failed")
                        .context(Code::from(err.code)));
                }
            }
            Ok(_) => bail!("unexpected StorageResponse.kind during initialization"),
            Err(err) => {
                return Err(err).context("failed to decode StorageResponse");
            }
        }

        Ok((session, server_tx, responses))
    }

    /// Converts an `oak_proto_rust` session request to a `kms_proto` request.
    fn convert_request(
        msg: &oak_proto_rust::oak::session::v1::SessionRequest,
    ) -> Result<SessionRequest> {
        SessionRequest::decode(msg.encode_to_vec().as_slice()).context("failed to convert request")
    }

    /// Converts an `oak_proto_rust` session response to a `kms_proto` response.
    fn convert_response(
        msg: &session_v1_service_proto::session_proto::oak::session::v1::SessionResponse,
    ) -> Result<SessionResponse> {
        SessionResponse::decode(msg.encode_to_vec().as_slice())
            .context("failed to convert response")
    }
}
