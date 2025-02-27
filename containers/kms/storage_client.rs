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

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, ensure, Context, Result};
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
pub struct StorageClient {
    sender: mpsc::UnboundedSender<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    _handle: AbortOnDropHandle<()>,
}

impl StorageClient {
    pub fn new<A, E, S>(
        client: OakSessionV1ServiceClient<tonic::transport::Channel>,
        init_fn: impl Fn() -> UpdateRequest + Send + 'static,
        attester: A,
        endorser: E,
        signer: S,
        reference_values: ReferenceValues,
    ) -> Self
    where
        A: Attester + Clone + 'static,
        E: Endorser + Clone + 'static,
        S: Signer + Clone + 'static,
    {
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
                clock: Arc::new(SystemClock {}),
                pending_requests: HashMap::new(),
                next_correlation_id: 1,
            }
            .run()
            .await
        });
        Self { sender, _handle: AbortOnDropHandle::new(handle) }
    }

    /// Performs a read operation.
    pub async fn read(&self, request: ReadRequest) -> Result<ReadResponse> {
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

    /// Performs an update operation.
    pub async fn update(&self, request: UpdateRequest) -> Result<UpdateResponse> {
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

struct MessageSender<InitFn, A, E, S> {
    tx: mpsc::UnboundedSender<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    rx: mpsc::UnboundedReceiver<(StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    client: OakSessionV1ServiceClient<tonic::transport::Channel>,
    init_fn: InitFn,
    pending_requests: HashMap<u64, (StorageRequest, oneshot::Sender<Result<StorageResponse>>)>,
    next_correlation_id: u64,
    attester: A,
    endorser: E,
    signer: S,
    reference_values: ReferenceValues,
    clock: Arc<SystemClock>,
}

impl<InitFn, A, E, S> MessageSender<InitFn, A, E, S>
where
    InitFn: Fn() -> UpdateRequest + 'static,
    A: Attester + Clone + 'static,
    E: Endorser + Clone + 'static,
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
        let mut session = create_session_config(
            Box::new(self.attester.clone()),
            Box::new(self.endorser.clone()),
            Box::new(self.signer.clone()),
            self.reference_values.clone(),
            self.clock.clone(),
        )
        .and_then(ClientSession::create)
        .context("failed to create ClientSession")?;
        let (server_tx, rx) = mpsc::channel(1);
        let mut responses = self.client.stream(ReceiverStream::new(rx)).await?.into_inner();
        let (init_tx, mut init_rx) = mpsc::channel(1);
        init_tx
            .send(StorageRequest {
                correlation_id: 0,
                kind: Some(storage_request::Kind::Update((self.init_fn)())),
            })
            .await
            .context("failed to enqueue init request")?;
        let mut initialized = false;

        // Send the initial message. Unlike all other steps in the protocol, the
        // ClientSession will return an error if `get_outgoing_message()` is
        // called an extra time before a response is received from the server.
        let msg = session
            .get_outgoing_message()?
            .ok_or_else(|| anyhow!("failed to get first message from ClientSession"))?;
        let msg = SessionRequest::decode(msg.encode_to_vec().as_slice())
            .context("failed to convert request")?;
        server_tx.send(msg).await?;

        loop {
            // Check for an incoming message from the server. Once initialization is
            // complete, also check for messages from the mpsc channel.
            select! {
                event = responses.message() => {
                    // Pass messages from the server to the ClientSession. Assuming they aren't part
                    // of the initial handshake, they'll be retrieved using `read` below.
                    match event? {
                        Some(msg) => {
                            let msg = SessionResponse::decode(msg.encode_to_vec().as_slice())
                                .context("failed to convert response")?;
                            session.put_incoming_message(&msg)?;
                        }
                        None => bail!("server unexpectedly closed stream"),
                    }
                }
                event = init_rx.recv(), if session.is_open() => {
                    // Once the session is open, write the initialization request to the
                    // ClientSession. It'll be retrieved using `get_outgoing_message` below.
                    session.write(&PlaintextMessage { plaintext: event.unwrap().encode_to_vec() })?;
                }
                event = self.rx.recv(), if initialized => {
                    // Once initialized, write messages from the mpsc channel to the ClientSession.
                    // They'll be retrieved using `get_outgoing_message` below.
                    match event {
                        Some((mut msg, tx)) => {
                            // Skip requests that have been abandoned.
                            if !tx.is_closed() {
                                msg.correlation_id = self.next_correlation_id;
                                self.next_correlation_id = self.next_correlation_id.wrapping_add(1);
                                session.write(
                                    &PlaintextMessage { plaintext: msg.encode_to_vec() }
                                )?;
                                self.pending_requests.insert(msg.correlation_id, (msg, tx));
                            }
                        }
                        None => { bail!("StorageClient stream unexpectedly closed") }
                    }
                }
            }

            // Send all already encrypted messages to the server. This also
            // includes messages for the initial handshake.
            while let Some(msg) = session.get_outgoing_message()? {
                let msg = SessionRequest::decode(msg.encode_to_vec().as_slice())
                    .context("failed to convert request")?;
                server_tx.send(msg).await?;
            }

            if session.is_open() {
                // Read any pending responses from the ClientSession.
                while let Some(msg) = session.read()? {
                    let msg = StorageResponse::decode(msg.plaintext.as_slice())
                        .context("failed to decode response")?;
                    if initialized {
                        if let Some((_, tx)) = self.pending_requests.remove(&msg.correlation_id) {
                            if let Err(err) = tx.send(Ok(msg)) {
                                warn!("failed to send response to client: {:?}", err);
                            }
                        } else {
                            warn!("unexpected correlation id in response: {}", msg.correlation_id);
                        }
                    } else {
                        // Process the initialization response. Initialization should only be
                        // performed once per cluster of storage servers, so this operation will
                        // frequently fail with a FAILED_PRECONDITION error.
                        ensure!(
                            msg.correlation_id == 0,
                            "unexpected correlation id in init response: {}",
                            msg.correlation_id,
                        );
                        match msg.kind {
                            Some(storage_response::Kind::Update(_)) => {
                                info!("Storage initialized!");
                            }
                            Some(storage_response::Kind::Error(err)) => {
                                if err.code == Code::FailedPrecondition as i32 {
                                    info!("Storage already initialized");
                                } else {
                                    return Err(anyhow::Error::msg(err.message)
                                        .context("storage initialization failed")
                                        .context(Code::from(err.code)));
                                }
                            }
                            _ => bail!("unexpected StorageResponse.kind during initialization"),
                        }
                        initialized = true;
                    }
                }
            }
        }
    }
}

struct SystemClock {}
impl Clock for SystemClock {
    fn get_milliseconds_since_epoch(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime before Unix epoch")
            .as_millis()
            .try_into()
            .expect("SystemTime too large")
    }
}
