// Copyright 2024 Google LLC.
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

use crate::replication::{ledger_event, ledger_event::*, LedgerEvent, LedgerSnapshot};
use crate::LedgerService;

use alloc::boxed::Box;
use federated_compute::proto::{
    ledger_request::*, ledger_response::*, Ledger, LedgerConfig, LedgerRequest, LedgerResponse,
    Status,
};
use oak_crypto::signer::Signer;
use prost::{bytes::Bytes, Message};
use slog::{debug, error, warn};
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};

pub struct LedgerActor {
    context: Option<Box<dyn ActorContext>>,
    ledger: LedgerService,
}

impl LedgerActor {
    pub fn new(signer: Box<dyn Signer>) -> Self {
        LedgerActor { context: None, ledger: LedgerService::new(signer) }
    }

    fn get_context(&mut self) -> &mut dyn ActorContext {
        self.context.as_mut().expect("Context is initialized").as_mut()
    }

    fn mut_ledger(&mut self) -> &mut LedgerService {
        &mut self.ledger
    }

    // Handles the actor message and returns the message outcome or the status to be
    // promptly returned to the untrusted side.
    fn handle_command(
        &mut self,
        command: ActorCommand,
    ) -> Result<CommandOutcome, micro_rpc::Status> {
        let ledger_request = LedgerRequest::decode(command.header.clone()).map_err(|error| {
            warn!(self.get_context().logger(), "LedgerActor: request cannot be parsed: {}", error);
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "LedgerRequest cannot be parsed",
            )
        })?;

        debug!(
            self.get_context().logger(),
            "LedgerActor: handling {} command",
            request_name(&ledger_request)
        );

        if !self.get_context().leader() {
            // Not a leader.
            warn!(
                self.get_context().logger(),
                "LedgerActor: command {} rejected: not a leader",
                request_name(&ledger_request)
            );
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Aborted,
                "Command rejected",
            ));
        }

        let event = match ledger_request.request {
            Some(Request::AuthorizeAccess(authorize_access_request)) => {
                // Attest and produce the event that contains all the data necessary to
                // update the budget and rewrap the symmetric key when the event is later
                // applied.
                let authorize_access_event = self
                    .mut_ledger()
                    .attest_and_produce_authorize_access_event(authorize_access_request)?;
                Event::AuthorizeAccess(authorize_access_event)
            }
            Some(Request::CreateKey(create_key_request)) => {
                // Produce the event that contains the pregenerate public/private key pair.
                let create_key_event =
                    self.mut_ledger().produce_create_key_event(create_key_request)?;
                Event::CreateKey(create_key_event)
            }
            Some(Request::DeleteKey(delete_key_request)) => {
                // In this case the original request is replicated as the event.
                Event::DeleteKey(delete_key_request)
            }
            Some(Request::RevokeAccess(revoke_access_request)) => {
                // In this case the original request is replicated as the event.
                Event::RevokeAccess(revoke_access_request)
            }
            _ => {
                warn!(
                    self.get_context().logger(),
                    "LedgerActor: unknown request {:?}", ledger_request
                );
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "LedgerActor: unexpected request type",
                ));
            }
        };

        Ok(CommandOutcome::with_event(ActorEvent::with_proto(
            command.correlation_id,
            &LedgerEvent { event: Some(event) },
        )))
    }

    fn handle_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, micro_rpc::Status> {
        let ledger_event = LedgerEvent::decode(event.contents.clone()).map_err(|error| {
            warn!(self.get_context().logger(), "LedgerActor: event cannot be parsed: {}", error);
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "LedgerRequest cannot be parsed",
            )
        })?;

        debug!(
            self.get_context().logger(),
            "LedgerActor: handling event at index {}: {}",
            context.index,
            event_name(&ledger_event)
        );

        let response = match ledger_event.event {
            Some(Event::AuthorizeAccess(authorize_access_event)) => {
                let authorize_access_response =
                    self.mut_ledger().apply_authorize_access_event(authorize_access_event)?;
                if !context.owned {
                    return Ok(EventOutcome::with_none());
                }
                Response::AuthorizeAccess(authorize_access_response)
            }
            Some(Event::CreateKey(create_key_event)) => {
                let create_key_response =
                    self.mut_ledger().apply_create_key_event(create_key_event)?;
                if !context.owned {
                    return Ok(EventOutcome::with_none());
                }
                Response::CreateKey(create_key_response)
            }
            Some(Event::DeleteKey(delete_key_request)) => {
                let delete_key_response = self.mut_ledger().delete_key(delete_key_request)?;
                if !context.owned {
                    return Ok(EventOutcome::with_none());
                }
                Response::DeleteKey(delete_key_response)
            }
            Some(ledger_event::Event::RevokeAccess(revoke_access_request)) => {
                let revoke_access_response =
                    self.mut_ledger().revoke_access(revoke_access_request)?;
                if !context.owned {
                    return Ok(EventOutcome::with_none());
                }
                Response::RevokeAccess(revoke_access_response)
            }
            _ => {
                warn!(self.get_context().logger(), "LedgerActor: unknown event {:?}", ledger_event);
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "LedgerActor: unexpected event type",
                ));
            }
        };

        Ok(EventOutcome::with_command(ActorCommand::with_header(
            event.correlation_id,
            &LedgerResponse { response: Some(response) },
        )))
    }
}

fn request_name(request: &LedgerRequest) -> &'static str {
    match request.request {
        Some(Request::AuthorizeAccess(_)) => "AuthorizeAccess",
        Some(Request::CreateKey(_)) => "CreateKey",
        Some(Request::DeleteKey(_)) => "DeleteKey",
        Some(Request::RevokeAccess(_)) => "RevokeAccess",
        _ => "Unknown",
    }
}

fn event_name(event: &LedgerEvent) -> &'static str {
    match event.event {
        Some(Event::AuthorizeAccess(_)) => "AuthorizeAccess",
        Some(Event::CreateKey(_)) => "CreateKey",
        Some(Event::DeleteKey(_)) => "DeleteKey",
        Some(Event::RevokeAccess(_)) => "RevokeAccess",
        _ => "Unknown",
    }
}

fn response_with_error(error: micro_rpc::Status) -> LedgerResponse {
    LedgerResponse {
        response: Some(Response::Error(Status {
            code: error.code as i32,
            message: error.message.into(),
        })),
    }
}

impl Actor for LedgerActor {
    /// Handles actor initialization. If error is returned the actor is
    /// considered in unknown state and is destroyed.
    fn on_init(&mut self, context: Box<dyn ActorContext>) -> Result<(), ActorError> {
        if self.context.is_some() {
            error!(self.get_context().logger(), "LedgerActor: already initialized");
            return Err(ActorError::Internal);
        }

        self.context = Some(context);
        debug!(self.get_context().logger(), "LedgerActor: initializing");

        let _ = LedgerConfig::decode(self.get_context().config().as_ref())
            .map_err(|_| ActorError::ConfigLoading)?;
        // TODO: use the config.

        Ok(())
    }

    /// Handles actor shutdown. After this method call completes the actor
    /// is destroyed.
    fn on_shutdown(&mut self) {}

    /// Handles creation of the actor state snapshot. If error is returned the
    /// actor is considered is unknown state and is destroyed.
    fn on_save_snapshot(&mut self) -> Result<Bytes, ActorError> {
        debug!(self.get_context().logger(), "LedgerActor: saving snapshot");
        let snapshot = self.mut_ledger().save_snapshot().map_err(|error| {
            error!(self.get_context().logger(), "LedgerActor: failed to save snapshot: {}", error);
            ActorError::Internal
        })?;
        Ok(snapshot.encode_to_vec().into())
    }

    /// Handles restoration of the actor state from snapshot. If error is
    /// returned the actor is considered is unknown state and is destroyed.
    fn on_load_snapshot(&mut self, snapshot: Bytes) -> Result<(), ActorError> {
        debug!(self.get_context().logger(), "LedgerActor: loading snapshot");
        let snapshot = LedgerSnapshot::decode(snapshot).map_err(|error| {
            error!(
                self.get_context().logger(),
                "LedgerActor: failed to decode snapshot: {}", error
            );
            ActorError::SnapshotLoading
        })?;
        self.mut_ledger().load_snapshot(snapshot).map_err(|error| {
            error!(self.get_context().logger(), "LedgerActor: failed to load snapshot: {}", error);
            ActorError::SnapshotLoading
        })?;
        Ok(())
    }

    /// Handles processing of a command by the actor. Command represents an
    /// intent of a consumer (e.g. request to update actor state). The
    /// command processing logic may decide to immediately respond (e.g. the
    /// command validation failed and cannot be executed) or to propose an
    /// event for replication by the consensus module (e.g. the
    /// event to update actor state once replicated).
    fn on_process_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Result<CommandOutcome, ActorError> {
        if command.is_none() {
            return Ok(CommandOutcome::with_none());
        }
        let command = command.unwrap();
        let correlation_id = command.correlation_id;
        self.handle_command(command).or_else(|err| {
            Ok(CommandOutcome::with_command(ActorCommand::with_header(
                correlation_id,
                &response_with_error(err),
            )))
        })
    }

    /// Handles committed events by applying them to the actor state. Event
    /// represents a state transition of the actor and may result in
    /// messages being sent to the consumer (e.g. response to the command
    /// that generated this event).
    fn on_apply_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, ActorError> {
        let correlation_id: u64 = event.correlation_id;
        self.handle_event(context, event).or_else(|err| {
            Ok(EventOutcome::with_command(ActorCommand::with_header(
                correlation_id,
                &response_with_error(err),
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oak_restricted_kernel_sdk::testing::MockSigner;
    use slog::Logger;
    use tcp_runtime::logger::log::create_logger;
    struct MockActorContext {
        logger: Logger,
    }

    impl MockActorContext {
        fn new() -> MockActorContext {
            MockActorContext { logger: create_logger() }
        }
    }

    impl ActorContext for MockActorContext {
        fn logger(&self) -> &Logger {
            &self.logger
        }
        fn id(&self) -> u64 {
            0u64
        }
        fn instant(&self) -> u64 {
            0u64
        }
        fn config(&self) -> Bytes {
            LedgerConfig {}.encode_to_vec().into()
        }
        fn leader(&self) -> bool {
            true
        }
    }

    fn create_actor() -> LedgerActor {
        let mock_context = Box::new(MockActorContext::new());
        let mut actor = LedgerActor::new(Box::new(MockSigner::create().unwrap()));
        assert_eq!(actor.on_init(mock_context), Ok(()));
        actor
    }

    #[test]
    fn test_create_actor() {
        let mut actor = create_actor();
        assert_eq!(actor.get_context().id(), 0u64);
    }

    #[test]
    fn test_save_snapshot() {
        let mut actor = create_actor();
        let snapshot = LedgerSnapshot {
            current_time: Some(prost_types::Timestamp::default()),
            ..Default::default()
        };
        assert_eq!(
            actor.on_save_snapshot().unwrap(),
            Into::<Bytes>::into(snapshot.encode_to_vec())
        );
    }

    #[test]
    fn test_load_snapshot() {
        let mut actor = create_actor();
        let snapshot = LedgerSnapshot {
            current_time: Some(prost_types::Timestamp::default()),
            ..Default::default()
        };
        assert_eq!(actor.on_load_snapshot(snapshot.encode_to_vec().into()), Ok(()));
    }
}
