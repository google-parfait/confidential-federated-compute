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

use anyhow::{anyhow, ensure, Context};
use hashbrown::{hash_map, HashMap};
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_crypto::signer::Signer;
use oak_proto_rust::oak::{
    attestation::v1::ReferenceValues,
    session::v1::{PlaintextMessage, SessionRequestWithSessionId, SessionResponse},
};
use oak_session::{config::SessionConfig, ProtocolEngine, ServerSession, Session};
use prost::{bytes::Bytes, Message};
use session_config::create_session_config;
use slog::{debug, error, warn};
use storage::Storage;
use storage_proto::confidential_federated_compute::kms::{
    read_request, storage_event, storage_request, storage_response, update_request, ReadRequest,
    ReadResponse, SessionResponseWithStatus, StorageEvent, StorageRequest, StorageResponse,
    UpdateRequest,
};
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};
use tonic::Code;

/// Actor that implements the Storage service.
///
/// This class accepts requests over a Noise channel and forwards them to the
/// Storage implementation.
pub struct StorageActor {
    // The underlying storage struct.
    storage: Storage,
    // A factory function that creates new configs for encrypted sessions.
    session_config_factory: Box<dyn Fn() -> anyhow::Result<SessionConfig>>,
    // The reference values used for attestation.
    reference_values: ReferenceValues,
    // The set of active encrypted Oak sessions, keyed by session ID.
    sessions: HashMap<Vec<u8>, ServerSession>,
    // The actor's context, populated during initialization.
    context: Option<Box<dyn ActorContext>>,
}

/// The outcome of a `StorageActor::handle_session_request` call.
enum HandleSessionRequestOutcome {
    None,
    Response(SessionResponse),
    Event(StorageEvent),
}

/// The outcome of a `StorageActor::handle_storage_request` call.
enum HandleStorageRequestOutcome {
    Response(StorageResponse),
    Event(StorageEvent),
}

impl StorageActor {
    pub fn new<A, E, S>(
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
        let storage = Storage::default();

        let rv_copy = reference_values.clone();
        let clock = storage.clock();
        let session_config_factory = Box::new(move || {
            create_session_config(
                Box::new(attester.clone()),
                Box::new(endorser.clone()),
                Box::new(signer.clone()),
                rv_copy.clone(),
                clock.clone(),
            )
        });

        Self {
            storage,
            session_config_factory,
            reference_values,
            sessions: HashMap::new(),
            context: None,
        }
    }

    fn logger(&self) -> &slog::Logger {
        self.context.as_ref().expect("StorageActor not initialized").logger()
    }

    /// Handles the decryption and processing of a SessionRequestWithSessionId
    /// message.
    fn handle_session_request(
        &mut self,
        session_request: SessionRequestWithSessionId,
    ) -> anyhow::Result<HandleSessionRequestOutcome> {
        // Requests should only be processed by the leader.
        if !self.context.as_deref().map(ActorContext::leader).unwrap_or(false) {
            warn!(self.logger(), "rejecting command: not a leader");
            return Err(anyhow!("rejecting command: not a leader").context(Code::Aborted));
        }

        // Look up (or create) the session. If the request is unset, the session is
        // complete.
        if session_request.request.is_none() {
            self.sessions.remove(&session_request.session_id);
            return Ok(HandleSessionRequestOutcome::Response(SessionResponse::default()));
        }
        let session = match self.sessions.entry_ref(session_request.session_id.as_slice()) {
            hash_map::EntryRef::Occupied(entry) => entry.into_mut(),
            hash_map::EntryRef::Vacant(entry) => entry.insert(
                (self.session_config_factory)()
                    .and_then(ServerSession::create)
                    .context("failed to create ServerSession")
                    .context(ActorError::Internal)?,
            ),
        };

        // Add the request to the ServerSession for decryption. Assuming the handshake
        // is complete, the resulting decrypted message will be read by `read()`
        // below.
        session
            .put_incoming_message(&session_request.request.unwrap())
            .context("SessionRequest is invalid")
            .context(Code::InvalidArgument)?;

        // If the initial handshake is complete, process the decrypted message (if any).
        if session.is_open() {
            let msg = session
                .read()
                .context("failed to read from session")?
                .ok_or_else(|| anyhow!("no message read from ServerSession"))?;
            let request = StorageRequest::decode(msg.plaintext.as_slice())
                .context("failed to decode StorageRequest")
                .context(Code::InvalidArgument)?;
            match Self::handle_storage_request(
                &mut self.storage,
                session_request.session_id,
                request,
            )? {
                HandleStorageRequestOutcome::Response(response) => {
                    // Encrypt the response by adding it back to the session. It'll be retrieved by
                    // `get_outgoing_message()` below.
                    session
                        .write(&PlaintextMessage { plaintext: response.encode_to_vec() })
                        .context("failed to write to session")?;
                }
                HandleStorageRequestOutcome::Event(event) => {
                    return Ok(HandleSessionRequestOutcome::Event(event));
                }
            }
            ensure!(
                session.read().context("failed to read from session")?.is_none(),
                "unexpected extra message"
            );
        }

        // Retrieve the encrypted response (if any) that should be sent to the client.
        // There may be no response during the initial handshake.
        let response = session.get_outgoing_message().context("failed to get outgoing message")?;
        if response.is_none() {
            return Ok(HandleSessionRequestOutcome::None);
        }
        ensure!(
            session.get_outgoing_message().context("failed to get outgoing message")?.is_none(),
            "unexpected extra outgoing message"
        );
        Ok(HandleSessionRequestOutcome::Response(response.unwrap()))
    }

    /// Handles the processing of a StorageRequest message.
    fn handle_storage_request(
        storage: &mut Storage,
        session_id: Vec<u8>,
        request: StorageRequest,
    ) -> anyhow::Result<HandleStorageRequestOutcome> {
        match request.kind {
            Some(storage_request::Kind::Read(msg)) => {
                // Read requests can be processed immediately by the leader.
                let response = storage.read(&msg)?;
                Ok(HandleStorageRequestOutcome::Response(StorageResponse {
                    correlation_id: request.correlation_id,
                    kind: Some(storage_response::Kind::Read(response)),
                }))
            }
            Some(storage_request::Kind::Update(msg)) => {
                // Convert the update into an event to be propagated to followers.
                Ok(HandleStorageRequestOutcome::Event(StorageEvent {
                    session_id: session_id,
                    correlation_id: request.correlation_id,
                    kind: Some(storage_event::Kind::Update(msg)),
                }))
            }
            _ => Err(anyhow!("unsupported StorageRequest.kind").context(Code::InvalidArgument)),
        }
    }

    fn handle_storage_event(
        &mut self,
        context: &ActorEventContext,
        event: StorageEvent,
    ) -> anyhow::Result<SessionResponse> {
        // Apply the event.
        let response_kind = match event.kind {
            Some(storage_event::Kind::Update(update)) => {
                self.storage.update(update).map(storage_response::Kind::Update)
            }
            _ => return Err(anyhow!("unsupported StorageEvent.kind").context(ActorError::Internal)),
        }
        .unwrap_or_else(|err| storage_response::Kind::Error(Self::convert_error(err)));

        // Only send a response if this actor generated the event. The actual
        // SessionResponse returned doesn't matter because `on_apply_event` will
        // return `EventOutcome::with_none()` in this case.
        if !context.owned {
            return Ok(SessionResponse::default());
        }

        // Write the response to the session.
        let session = self
            .sessions
            .get_mut(&event.session_id)
            .ok_or_else(|| anyhow!("session not found").context(Code::NotFound))?;
        ensure!(session.is_open());
        let response =
            StorageResponse { correlation_id: event.correlation_id, kind: Some(response_kind) };
        session
            .write(&PlaintextMessage { plaintext: response.encode_to_vec() })
            .context("failed to write to session")?;

        if let Some(msg) = session.get_outgoing_message()? {
            ensure!(session.get_outgoing_message()?.is_none(), "unexpected extra outgoing message");
            Ok(msg)
        } else {
            Err(anyhow!("missing outgoing message"))
        }
    }

    fn convert_error(err: anyhow::Error) -> storage_proto::status_proto::google::rpc::Status {
        storage_proto::status_proto::google::rpc::Status {
            code: (err.downcast_ref::<Code>().copied().unwrap_or(Code::Internal)).into(),
            message: format!("{err:#}"), // Include causes in the message.
            ..Default::default()
        }
    }
}

impl Actor for StorageActor {
    fn get_reference_values(&self) -> ReferenceValues {
        self.reference_values.clone()
    }

    fn on_init(&mut self, context: Box<dyn ActorContext>) -> Result<(), ActorError> {
        debug!(context.logger(), "StorageActor::on_init");
        if self.context.is_some() {
            error!(self.logger(), "StorageActor already initialized");
            return Err(ActorError::Internal);
        }

        self.context = Some(context);
        Ok(())
    }

    fn on_shutdown(&mut self) {
        debug!(self.logger(), "StorageActor::on_shutdown");
    }

    fn on_save_snapshot(&mut self) -> Result<Bytes, ActorError> {
        debug!(self.logger(), "StorageActor::on_save_snapshot");
        // Use a ReadResponse for the full key space as a snapshot.
        let response = self
            .storage
            .read(&ReadRequest {
                ranges: vec![read_request::Range {
                    start: u128::MIN.to_be_bytes().to_vec(),
                    end: Some(u128::MAX.to_be_bytes().to_vec()),
                }],
            })
            .map_err(|err| {
                error!(self.logger(), "Failed to generate snapshot: {:?}", err);
                ActorError::Internal
            })?;
        Ok(response.encode_to_vec().into())
    }

    fn on_load_snapshot(&mut self, snapshot: Bytes) -> Result<(), ActorError> {
        debug!(self.logger(), "StorageActor::on_load_snapshot");
        let snapshot = ReadResponse::decode(snapshot).map_err(|err| {
            error!(self.logger(), "Failed to decode snapshot: {}", err);
            ActorError::SnapshotLoading
        })?;

        // Load the entries from the snapshot.
        self.storage = Storage::default();
        self.storage
            .update(UpdateRequest {
                now: snapshot.now,
                updates: snapshot
                    .entries
                    .into_iter()
                    .map(|entry| update_request::Update {
                        key: entry.key,
                        value: Some(entry.value),
                    })
                    .collect(),
            })
            .map_err(|err| {
                error!(self.logger(), "Failed to load snapshot: {:?}", err);
                ActorError::SnapshotLoading
            })?;
        Ok(())
    }

    fn on_process_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Result<CommandOutcome, ActorError> {
        if command.is_none() {
            // TODO: b/393146003 - Periodically send an empty UpdateRequest to
            // ensure that the clock is regularly advanced.
            return Ok(CommandOutcome::with_none());
        }
        let command = command.unwrap();
        debug!(self.logger(), "StorageActor::on_process_command: {}", command.correlation_id);

        let outcome = || -> anyhow::Result<CommandOutcome> {
            let request = SessionRequestWithSessionId::decode(command.header)
                .context("failed to decode SessionRequestWithSessionId")
                .context(Code::InvalidArgument)?;

            match self.handle_session_request(request)? {
                HandleSessionRequestOutcome::None => {
                    Ok(CommandOutcome::with_command(ActorCommand::with_header(
                        command.correlation_id,
                        &SessionResponseWithStatus::default(),
                    )))
                }
                HandleSessionRequestOutcome::Response(response) => {
                    // Until Oak uses `rust_prost_library`, the types in `oak_proto_rust` are
                    // not the same as the types in `storage_proto`.
                    use storage_proto::session_proto::oak::session::v1::SessionResponse;
                    let response = SessionResponse::decode(response.encode_to_vec().as_slice())?;
                    Ok(CommandOutcome::with_command(ActorCommand::with_header(
                        command.correlation_id,
                        &SessionResponseWithStatus {
                            response: Some(response),
                            ..Default::default()
                        },
                    )))
                }
                HandleSessionRequestOutcome::Event(event) => Ok(CommandOutcome::with_event(
                    ActorEvent::with_proto(command.correlation_id, &event),
                )),
            }
        }();

        outcome.or_else(|err| {
            // If there's an ActorError attached, return it.
            if let Some(actor_error) = err.downcast_ref::<ActorError>() {
                error!(self.logger(), "Failed to handle ActorCommand: {:?}", err);
                return Err(actor_error.clone());
            }

            // Return application-level errors as part of the response.
            Ok(CommandOutcome::with_command(ActorCommand::with_header(
                command.correlation_id,
                &SessionResponseWithStatus {
                    status: Some(Self::convert_error(err)),
                    ..Default::default()
                },
            )))
        })
    }

    fn on_apply_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, ActorError> {
        debug!(self.logger(), "StorageActor::on_apply_event: {}", event.correlation_id);
        let outcome = || -> anyhow::Result<EventOutcome> {
            let storage_event = StorageEvent::decode(event.contents)
                .context("failed to decode StorageEvent")
                .context(ActorError::Internal)?;

            let response = self.handle_storage_event(&context, storage_event)?;

            // A response should only be returned if this actor originally produced the
            // event.
            if !context.owned {
                return Ok(EventOutcome::with_none());
            }

            // Until Oak uses `rust_prost_library`, the types in `oak_proto_rust` are not
            // the same as the types in `storage_proto`.
            use storage_proto::session_proto::oak::session::v1::SessionResponse;
            let response = SessionResponse::decode(response.encode_to_vec().as_slice())?;
            Ok(EventOutcome::with_command(ActorCommand::with_header(
                event.correlation_id,
                &SessionResponseWithStatus { response: Some(response), ..Default::default() },
            )))
        }();

        outcome.or_else(|err| {
            // If there's an ActorError attached, return it.
            if let Some(actor_error) = err.downcast_ref::<ActorError>() {
                error!(self.logger(), "Failed to handle StorageEvent: {:?}", err);
                return Err(actor_error.clone());
            }

            // Otherwise, it's a application-level error, which should be wrapped in a
            // response. This application error should only be returned if this actor
            // originally produced the event.
            if !context.owned {
                return Ok(EventOutcome::with_none());
            }
            Ok(EventOutcome::with_command(ActorCommand::with_header(
                event.correlation_id,
                &SessionResponseWithStatus {
                    status: Some(Self::convert_error(err)),
                    ..Default::default()
                },
            )))
        })
    }
}
