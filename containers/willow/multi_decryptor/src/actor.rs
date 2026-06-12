// Copyright 2026 The Trusted Computations Platform Authors.
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

#![cfg_attr(not(feature = "std"), no_std)]
#![feature(never_type)]

extern crate alloc;
extern crate prost;
extern crate slog;
extern crate tcp_runtime;

use alloc::boxed::Box;
use alloc::rc::Rc;
use oak_proto_rust::oak::attestation::v1::ReferenceValues;
use prost::bytes::Bytes;
use prost::Message;
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};
use willow_committee_selector_service::actor::CommitteeSelectorActor;
use willow_committee_selector_service::apps::willow::committee_selector::service::{
    CommitteeSelectorConfig, CommitteeSelectorEvent, CommitteeSelectorResponse,
};
use willow_reputable_decryptor_service::actor::ReputableDecryptorActor;
use willow_reputable_decryptor_service::apps::willow::reputable_decryptor::service::{
    ReputableDecryptorConfig, ReputableDecryptorEvent, ReputableDecryptorResponse,
};

// Include generated multi_decryptor proto
pub mod multi_decryptor {
    pub mod service {
        include!(concat!(env!("OUT_DIR"), "/apps.willow.multi_decryptor.service.rs"));
    }
}

use crate::multi_decryptor::service::{
    multi_decryptor_event, multi_decryptor_request, multi_decryptor_response, MultiDecryptorConfig,
    MultiDecryptorEvent, MultiDecryptorRequest, MultiDecryptorResponse, MultiDecryptorSnapshot,
};

struct SharedActorContext {
    inner: Rc<Box<dyn ActorContext>>,
    config_bytes: Bytes,
}

impl ActorContext for SharedActorContext {
    fn logger(&self) -> &slog::Logger {
        self.inner.as_ref().logger()
    }

    fn id(&self) -> u64 {
        self.inner.as_ref().id()
    }

    fn instant(&self) -> u64 {
        self.inner.as_ref().instant()
    }

    fn config(&self) -> Bytes {
        self.config_bytes.clone()
    }

    fn leader(&self) -> bool {
        self.inner.as_ref().leader()
    }
}

pub struct MultiDecryptorActor {
    reference_values: ReferenceValues,
    committee_selector: Option<CommitteeSelectorActor>,
    reputable_decryptor: Option<ReputableDecryptorActor>,
}

impl MultiDecryptorActor {
    pub fn new(reference_values: ReferenceValues) -> Self {
        Self { reference_values, committee_selector: None, reputable_decryptor: None }
    }

    pub fn new_insecure() -> Self {
        Self::new(ReferenceValues::default())
    }

    fn wrap_command_cs(&self, cmd: ActorCommand) -> ActorCommand {
        let cs_response = CommitteeSelectorResponse::decode(cmd.header)
            .expect("failed to decode CommitteeSelectorResponse");
        let multi_decryptor_response = MultiDecryptorResponse {
            msg: Some(multi_decryptor_response::Msg::CommitteeSelector(cs_response)),
        };
        ActorCommand {
            correlation_id: cmd.correlation_id,
            header: multi_decryptor_response.encode_to_vec().into(),
            payload: cmd.payload,
        }
    }

    fn wrap_event_cs(&self, evt: ActorEvent) -> ActorEvent {
        let cs_event = CommitteeSelectorEvent::decode(evt.contents)
            .expect("failed to decode CommitteeSelectorEvent");
        let multi_decryptor_event = MultiDecryptorEvent {
            event: Some(multi_decryptor_event::Event::CommitteeSelector(cs_event)),
        };
        ActorEvent {
            correlation_id: evt.correlation_id,
            contents: multi_decryptor_event.encode_to_vec().into(),
        }
    }

    fn wrap_command_outcome_cs(&self, outcome: CommandOutcome) -> CommandOutcome {
        CommandOutcome {
            commands: outcome.commands.into_iter().map(|cmd| self.wrap_command_cs(cmd)).collect(),
            event: outcome.event.map(|evt| self.wrap_event_cs(evt)),
        }
    }

    fn wrap_event_outcome_cs(&self, outcome: EventOutcome) -> EventOutcome {
        EventOutcome {
            commands: outcome.commands.into_iter().map(|cmd| self.wrap_command_cs(cmd)).collect(),
        }
    }

    fn wrap_command_rd(&self, cmd: ActorCommand) -> ActorCommand {
        let rd_response = ReputableDecryptorResponse::decode(cmd.header)
            .expect("failed to decode ReputableDecryptorResponse");
        let multi_decryptor_response = MultiDecryptorResponse {
            msg: Some(multi_decryptor_response::Msg::ReputableDecryptor(rd_response)),
        };
        ActorCommand {
            correlation_id: cmd.correlation_id,
            header: multi_decryptor_response.encode_to_vec().into(),
            payload: cmd.payload,
        }
    }

    fn wrap_event_rd(&self, evt: ActorEvent) -> ActorEvent {
        let rd_event = ReputableDecryptorEvent::decode(evt.contents)
            .expect("failed to decode ReputableDecryptorEvent");
        let multi_decryptor_event = MultiDecryptorEvent {
            event: Some(multi_decryptor_event::Event::ReputableDecryptor(rd_event)),
        };
        ActorEvent {
            correlation_id: evt.correlation_id,
            contents: multi_decryptor_event.encode_to_vec().into(),
        }
    }

    fn wrap_command_outcome_rd(&self, outcome: CommandOutcome) -> CommandOutcome {
        CommandOutcome {
            commands: outcome.commands.into_iter().map(|cmd| self.wrap_command_rd(cmd)).collect(),
            event: outcome.event.map(|evt| self.wrap_event_rd(evt)),
        }
    }

    fn wrap_event_outcome_rd(&self, outcome: EventOutcome) -> EventOutcome {
        EventOutcome {
            commands: outcome.commands.into_iter().map(|cmd| self.wrap_command_rd(cmd)).collect(),
        }
    }
}

impl Actor for MultiDecryptorActor {
    fn on_init(&mut self, context: Box<dyn ActorContext>) -> Result<(), ActorError> {
        let config = MultiDecryptorConfig::decode(context.config().as_ref())
            .map_err(|_| ActorError::ConfigLoading)?;

        self.committee_selector =
            Some(CommitteeSelectorActor::new_with_reference_values(self.reference_values.clone()));
        self.reputable_decryptor =
            Some(ReputableDecryptorActor::new_with_reference_values(self.reference_values.clone()));

        let cs_config =
            CommitteeSelectorConfig { max_number_of_committees: config.max_number_of_committees };
        let mut cs_config_bytes = alloc::vec::Vec::new();
        cs_config.encode(&mut cs_config_bytes).unwrap();

        let rd_config = ReputableDecryptorConfig {
            max_number_of_decryptors: config.max_number_of_decryptors,
            max_number_of_keys: config.max_number_of_keys,
        };
        let mut rd_config_bytes = alloc::vec::Vec::new();
        rd_config.encode(&mut rd_config_bytes).unwrap();

        let shared = Rc::new(context);

        let ctx_cs = Box::new(SharedActorContext {
            inner: shared.clone(),
            config_bytes: cs_config_bytes.into(),
        });
        let ctx_rd =
            Box::new(SharedActorContext { inner: shared, config_bytes: rd_config_bytes.into() });

        self.committee_selector.as_mut().unwrap().on_init(ctx_cs)?;
        self.reputable_decryptor.as_mut().unwrap().on_init(ctx_rd)?;
        Ok(())
    }

    fn on_shutdown(&mut self) {
        if let Some(ref mut cs) = self.committee_selector {
            cs.on_shutdown();
        }
        if let Some(ref mut rd) = self.reputable_decryptor {
            rd.on_shutdown();
        }
    }

    fn on_save_snapshot(&mut self) -> Result<Bytes, ActorError> {
        let snapshot_cs_bytes =
            self.committee_selector.as_mut().ok_or(ActorError::Internal)?.on_save_snapshot()?;
        let snapshot_rd_bytes =
            self.reputable_decryptor.as_mut().ok_or(ActorError::Internal)?.on_save_snapshot()?;

        let multi_decryptor_snapshot = MultiDecryptorSnapshot {
            committee_selector: snapshot_cs_bytes,
            reputable_decryptor: snapshot_rd_bytes,
        };

        Ok(multi_decryptor_snapshot.encode_to_vec().into())
    }

    fn on_load_snapshot(&mut self, snapshot: Bytes) -> Result<(), ActorError> {
        let multi_decryptor_snapshot =
            MultiDecryptorSnapshot::decode(snapshot).map_err(|_| ActorError::SnapshotLoading)?;

        self.committee_selector
            .as_mut()
            .ok_or(ActorError::Internal)?
            .on_load_snapshot(multi_decryptor_snapshot.committee_selector)?;
        self.reputable_decryptor
            .as_mut()
            .ok_or(ActorError::Internal)?
            .on_load_snapshot(multi_decryptor_snapshot.reputable_decryptor)?;

        Ok(())
    }

    fn on_process_command(
        &mut self,
        command: Option<ActorCommand>,
    ) -> Result<CommandOutcome, ActorError> {
        if command.is_none() {
            return Ok(CommandOutcome::with_none());
        }
        let command = command.unwrap();
        let multi_decryptor_request = MultiDecryptorRequest::decode(command.header.clone())
            .map_err(|_| ActorError::Internal)?;

        match multi_decryptor_request.msg {
            Some(multi_decryptor_request::Msg::CommitteeSelector(req)) => {
                let inner_command = ActorCommand {
                    correlation_id: command.correlation_id,
                    header: req.encode_to_vec().into(),
                    payload: command.payload,
                };
                let outcome = self
                    .committee_selector
                    .as_mut()
                    .ok_or(ActorError::Internal)?
                    .on_process_command(Some(inner_command))?;
                Ok(self.wrap_command_outcome_cs(outcome))
            }
            Some(multi_decryptor_request::Msg::ReputableDecryptor(req)) => {
                let inner_command = ActorCommand {
                    correlation_id: command.correlation_id,
                    header: req.encode_to_vec().into(),
                    payload: command.payload,
                };
                let outcome = self
                    .reputable_decryptor
                    .as_mut()
                    .ok_or(ActorError::Internal)?
                    .on_process_command(Some(inner_command))?;
                Ok(self.wrap_command_outcome_rd(outcome))
            }
            None => Ok(CommandOutcome::with_none()),
        }
    }

    fn on_apply_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, ActorError> {
        let multi_decryptor_event = MultiDecryptorEvent::decode(event.contents.clone())
            .map_err(|_| ActorError::Internal)?;

        match multi_decryptor_event.event {
            Some(multi_decryptor_event::Event::CommitteeSelector(evt)) => {
                let inner_event = ActorEvent {
                    correlation_id: event.correlation_id,
                    contents: evt.encode_to_vec().into(),
                };
                let outcome = self
                    .committee_selector
                    .as_mut()
                    .ok_or(ActorError::Internal)?
                    .on_apply_event(context, inner_event)?;
                Ok(self.wrap_event_outcome_cs(outcome))
            }
            Some(multi_decryptor_event::Event::ReputableDecryptor(evt)) => {
                let inner_event = ActorEvent {
                    correlation_id: event.correlation_id,
                    contents: evt.encode_to_vec().into(),
                };
                let outcome = self
                    .reputable_decryptor
                    .as_mut()
                    .ok_or(ActorError::Internal)?
                    .on_apply_event(context, inner_event)?;
                Ok(self.wrap_event_outcome_rd(outcome))
            }
            None => Ok(EventOutcome::with_none()),
        }
    }

    fn get_reference_values(&self) -> ReferenceValues {
        self.reference_values.clone()
    }
}
