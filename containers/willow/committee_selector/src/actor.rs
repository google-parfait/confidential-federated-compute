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

use crate::apps::willow::committee_selector::service::{
    committee_selector_event, committee_selector_request, committee_selector_response,
    CheckCommitteeStatusRequest, CheckCommitteeStatusResponse, CommitteeSelectorConfig,
    CommitteeSelectorEvent, CommitteeSelectorRequest, CommitteeSelectorResponse,
    CommitteeSelectorSnapshot, CommitteeSelectorStatus, CreateCommitteeEvent,
    CreateCommitteeRequest, CreateCommitteeResponse, EndorsementStatus, SampleCommitteeEvent,
    SampleCommitteeRequest, VolunteerBatchForCommitteeEvent, VolunteerBatchForCommitteeRequest,
    VolunteerBatchForCommitteeResponse,
};
use crate::selector::{fingerprint, CommitteeSelector};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::{boxed::Box, vec::Vec};
use micro_rpc::StatusCode;
use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    ApplicationLayerReferenceValues, BinaryReferenceValue, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakRestrictedKernelReferenceValues,
    ReferenceValues, RootLayerReferenceValues, SkipVerification, TextReferenceValue,
};
use prost::{bytes::Bytes, Message};
use slog::{debug, warn};
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};

/// The system actor wrapping the `CommitteeSelector` and exposing it to the
/// Trusted Computations Platform (TCP) runtime.
///
/// Handlers map protobuf request commands (e.g., `CreateCommittee`,
/// `VolunteerBatchForCommittee`, `SampleCommittee`) into state mutations on the
/// underlying `CommitteeSelector`, and coordinate state machine persistence
/// by outputting consensus events to the replication logs.
pub struct CommitteeSelectorActor {
    reference_values: ReferenceValues,
    context: Option<Box<dyn ActorContext>>,
    selector: CommitteeSelector,
}

impl CommitteeSelectorActor {
    pub fn new() -> Self {
        let skip = BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
        };
        Self::new_with_reference_values(ReferenceValues {
            r#type: Some(reference_values::Type::OakRestrictedKernel(
                OakRestrictedKernelReferenceValues {
                    root_layer: Some(RootLayerReferenceValues {
                        insecure: Some(InsecureReferenceValues::default()),
                        ..Default::default()
                    }),
                    kernel_layer: Some(KernelLayerReferenceValues {
                        kernel: Some(KernelBinaryReferenceValue {
                            r#type: Some(kernel_binary_reference_value::Type::Skip(
                                SkipVerification::default(),
                            )),
                        }),
                        kernel_cmd_line_text: Some(TextReferenceValue {
                            r#type: Some(text_reference_value::Type::Skip(
                                SkipVerification::default(),
                            )),
                        }),
                        init_ram_fs: Some(skip.clone()),
                        memory_map: Some(skip.clone()),
                        acpi: Some(skip.clone()),
                        ..Default::default()
                    }),
                    application_layer: Some(ApplicationLayerReferenceValues {
                        binary: Some(skip.clone()),
                        configuration: Some(skip.clone()),
                    }),
                },
            )),
        })
    }

    pub fn new_with_reference_values(reference_values: ReferenceValues) -> Self {
        CommitteeSelectorActor {
            reference_values,
            context: None,
            selector: CommitteeSelector::new(0),
        }
    }

    fn get_context(&mut self) -> &mut dyn ActorContext {
        self.context.as_mut().expect("Context is initialized").as_mut()
    }

    fn process_create_committee_command(
        &mut self,
        correlation_id: u64,
        create_committee_request: CreateCommitteeRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let committee_id: i64 = create_committee_request.committee_id;

        if self.selector.committees.contains_key(&committee_id) {
            return Ok(CommandOutcome::with_command(ActorCommand::with_header(
                correlation_id,
                &CommitteeSelectorResponse {
                    msg: Some(committee_selector_response::Msg::CreateCommittee(
                        CreateCommitteeResponse {},
                    )),
                },
            )));
        }

        return Ok(CommandOutcome::with_event(ActorEvent::with_proto(
            correlation_id,
            &CommitteeSelectorEvent {
                event: Some(committee_selector_event::Event::CreateCommitteeEvent(
                    CreateCommitteeEvent { committee_id: committee_id.clone() },
                )),
            },
        )));
    }

    fn process_create_committee_event(
        &mut self,
        context: ActorEventContext,
        correlation_id: u64,
        create_committee_event: CreateCommitteeEvent,
    ) -> Result<EventOutcome, ActorError> {
        let committee_id = create_committee_event.committee_id;

        let logger = self.context.as_ref().expect("Context is initialized").logger();
        self.selector.create_committee(committee_id, logger);

        if context.owned {
            return Ok(EventOutcome::with_command(ActorCommand::with_header(
                correlation_id,
                &CommitteeSelectorResponse {
                    msg: Some(committee_selector_response::Msg::CreateCommittee(
                        CreateCommitteeResponse {},
                    )),
                },
            )));
        }

        Ok(EventOutcome::with_none())
    }

    fn process_volunteer_batch_for_committee_command(
        &mut self,
        correlation_id: u64,
        volunteer_batch_for_committee_request: VolunteerBatchForCommitteeRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let mut volunteer_key_digests = Vec::new();

        for volunteer_request in volunteer_batch_for_committee_request.volunteers {
            let public_key = &volunteer_request.public_key;
            let public_key_digest = fingerprint(public_key) as i64;
            let key_endorsement_status = EndorsementStatus::Valid;

            volunteer_key_digests.push(
                crate::apps::willow::committee_selector::service::VolunteerKeyDigest {
                    public_key_digest,
                    key_endorsement_status: key_endorsement_status.into(),
                },
            );
        }

        let mut rng_data = Vec::new();
        rng_data.extend_from_slice(&rand::random::<u64>().to_le_bytes());
        rng_data.extend_from_slice(&correlation_id.to_le_bytes());
        let randomness = fingerprint(&rng_data);

        Ok(CommandOutcome::with_event(ActorEvent::with_proto(
            correlation_id,
            &CommitteeSelectorEvent {
                event: Some(committee_selector_event::Event::VolunteerBatchForCommitteeEvent(
                    VolunteerBatchForCommitteeEvent { volunteer_key_digests, randomness },
                )),
            },
        )))
    }

    fn process_volunteer_batch_for_committee_event(
        &mut self,
        context: ActorEventContext,
        correlation_id: u64,
        volunteer_batch_for_committee_event: VolunteerBatchForCommitteeEvent,
    ) -> Result<EventOutcome, ActorError> {
        let assignments = self.selector.volunteer_batch(
            volunteer_batch_for_committee_event.volunteer_key_digests,
            volunteer_batch_for_committee_event.randomness,
        );

        if context.owned {
            return Ok(EventOutcome::with_command(ActorCommand::with_header(
                correlation_id,
                &CommitteeSelectorResponse {
                    msg: Some(committee_selector_response::Msg::VolunteerBatchForCommittee(
                        VolunteerBatchForCommitteeResponse { assignments },
                    )),
                },
            )));
        }

        Ok(EventOutcome::with_none())
    }

    fn process_sample_committee_command(
        &mut self,
        correlation_id: u64,
        sample_committee_request: SampleCommitteeRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let committee_id: i64 = sample_committee_request.committee_id;
        if !self.selector.committees.contains_key(&committee_id) {
            return self.command_err(
                correlation_id,
                StatusCode::FailedPrecondition as i32,
                format!("Committee not found for committee id {}", committee_id),
            );
        }

        return Ok(CommandOutcome::with_event(ActorEvent::with_proto(
            correlation_id,
            &CommitteeSelectorEvent {
                event: Some(committee_selector_event::Event::SampleCommitteeEvent(
                    SampleCommitteeEvent { committee_id: committee_id.clone() },
                )),
            },
        )));
    }

    fn process_sample_committee_event(
        &mut self,
        context: ActorEventContext,
        correlation_id: u64,
        sample_committee_event: SampleCommitteeEvent,
    ) -> Result<EventOutcome, ActorError> {
        let committee_id = sample_committee_event.committee_id;

        match self.selector.sample_committee(committee_id) {
            Ok((status, members, rejected_volunteers_count)) => {
                if context.owned {
                    return Ok(EventOutcome::with_command(ActorCommand::with_header(
                        correlation_id,
                        &CommitteeSelectorResponse {
                            msg: Some(committee_selector_response::Msg::CheckCommitteeStatus(
                                CheckCommitteeStatusResponse {
                                    committee_id,
                                    status: status.into(),
                                    members,
                                    rejected_volunteers_count: rejected_volunteers_count as i64,
                                },
                            )),
                        },
                    )));
                }
                Ok(EventOutcome::with_none())
            }
            Err(msg) => {
                if context.owned {
                    return self.event_err(
                        correlation_id,
                        StatusCode::FailedPrecondition as i32,
                        msg,
                    );
                }
                Ok(EventOutcome::with_none())
            }
        }
    }

    fn process_check_committee_status_command(
        &mut self,
        correlation_id: u64,
        check_committee_status_request: CheckCommitteeStatusRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let committee_id = check_committee_status_request.committee_id;

        match self.selector.check_committee_status(committee_id) {
            Some((status, members, rejected_volunteers_count)) => {
                Ok(CommandOutcome::with_command(ActorCommand::with_header(
                    correlation_id,
                    &CommitteeSelectorResponse {
                        msg: Some(committee_selector_response::Msg::CheckCommitteeStatus(
                            CheckCommitteeStatusResponse {
                                committee_id,
                                status: status.into(),
                                members,
                                rejected_volunteers_count: rejected_volunteers_count as i64,
                            },
                        )),
                    },
                )))
            }
            None => self.command_err(
                correlation_id,
                StatusCode::FailedPrecondition as i32,
                format!("Committee not found for given {} committee id", committee_id),
            ),
        }
    }

    fn command_err(
        &mut self,
        correlation_id: u64,
        code: i32,
        message: String,
    ) -> Result<CommandOutcome, ActorError> {
        return Ok(CommandOutcome::with_command(ActorCommand::with_header(
            correlation_id,
            &CommitteeSelectorResponse {
                msg: Some(committee_selector_response::Msg::Error(CommitteeSelectorStatus {
                    code: code,
                    message: message,
                })),
            },
        )));
    }

    fn event_err(
        &mut self,
        correlation_id: u64,
        code: i32,
        message: String,
    ) -> Result<EventOutcome, ActorError> {
        return Ok(EventOutcome::with_command(ActorCommand::with_header(
            correlation_id,
            &CommitteeSelectorResponse {
                msg: Some(committee_selector_response::Msg::Error(CommitteeSelectorStatus {
                    code: code,
                    message: message,
                })),
            },
        )));
    }
}

impl Actor for CommitteeSelectorActor {
    fn on_init(&mut self, context: Box<dyn ActorContext>) -> Result<(), ActorError> {
        self.context = Some(context);
        let config = CommitteeSelectorConfig::decode(self.get_context().config().as_ref())
            .map_err(|_| ActorError::ConfigLoading)?;
        self.selector.max_number_of_committees = config.max_number_of_committees as usize;
        Ok(())
    }

    fn on_shutdown(&mut self) {}

    fn on_save_snapshot(&mut self) -> Result<Bytes, ActorError> {
        let snapshot = self.selector.save_snapshot();
        Ok(snapshot.encode_to_vec().into())
    }

    fn on_load_snapshot(&mut self, snapshot: Bytes) -> Result<(), ActorError> {
        debug!(self.get_context().logger(), "Loading snapshot");

        let snapshot =
            CommitteeSelectorSnapshot::decode(snapshot).map_err(|_| ActorError::SnapshotLoading)?;

        self.selector.load_snapshot(snapshot).map_err(|_| ActorError::SnapshotLoading)?;

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

        let committee_selector_request =
            CommitteeSelectorRequest::decode(command.header).map_err(|error| {
                warn!(
                    self.get_context().logger(),
                    "CommitteeSelectorActor: request cannot be parsed: {}", error
                );
                ActorError::Internal
            })?;

        debug!(
            self.get_context().logger(),
            "CommitteeSelectorActor: handling {} command",
            request_name(&committee_selector_request)
        );

        if !self.get_context().leader() {
            warn!(
                self.get_context().logger(),
                "CommitteeSelectorActor: command {} rejected: not a leader",
                request_name(&committee_selector_request)
            );
            return self.command_err(
                command.correlation_id,
                StatusCode::FailedPrecondition as i32,
                "Node is not the leader".to_string(),
            );
        }

        match committee_selector_request.msg {
            Some(committee_selector_request::Msg::CreateCommittee(create_committee_request)) => {
                return self.process_create_committee_command(
                    command.correlation_id,
                    create_committee_request,
                );
            }
            Some(committee_selector_request::Msg::VolunteerBatchForCommittee(
                volunteer_batch_for_committee_request,
            )) => {
                return self.process_volunteer_batch_for_committee_command(
                    command.correlation_id,
                    volunteer_batch_for_committee_request,
                );
            }
            Some(committee_selector_request::Msg::SampleCommittee(sample_committee_request)) => {
                return self.process_sample_committee_command(
                    command.correlation_id,
                    sample_committee_request,
                );
            }
            Some(committee_selector_request::Msg::CheckCommitteeStatus(
                check_committee_status_request,
            )) => {
                return self.process_check_committee_status_command(
                    command.correlation_id,
                    check_committee_status_request,
                );
            }
            _ => {
                warn!(
                    self.get_context().logger(),
                    "CommitteeSelectorActor: unknown request {:?}", committee_selector_request
                );
                return self.command_err(
                    command.correlation_id,
                    StatusCode::InvalidArgument as i32,
                    format!(
                        "CommitteeSelectorActor: unknown request {:?}",
                        committee_selector_request
                    ),
                );
            }
        };
    }

    fn on_apply_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, ActorError> {
        let committee_selector_event =
            CommitteeSelectorEvent::decode(event.contents).map_err(|error| {
                warn!(
                    self.get_context().logger(),
                    "CommitteeSelectorActor: event cannot be parsed: {}", error
                );
                ActorError::Internal
            })?;

        debug!(
            self.get_context().logger(),
            "CommitteeSelectorActor: handling {} event",
            event_name(&committee_selector_event)
        );

        match committee_selector_event.event {
            Some(committee_selector_event::Event::CreateCommitteeEvent(create_committee_event)) => {
                return self.process_create_committee_event(
                    context,
                    event.correlation_id,
                    create_committee_event,
                );
            }
            Some(committee_selector_event::Event::VolunteerBatchForCommitteeEvent(
                volunteer_batch_for_committee_event,
            )) => {
                return self.process_volunteer_batch_for_committee_event(
                    context,
                    event.correlation_id,
                    volunteer_batch_for_committee_event,
                );
            }
            Some(committee_selector_event::Event::SampleCommitteeEvent(sample_committee_event)) => {
                return self.process_sample_committee_event(
                    context,
                    event.correlation_id,
                    sample_committee_event,
                );
            }
            _ => {
                warn!(
                    self.get_context().logger(),
                    "CommitteeSelectorActor: unknown event {:?}", committee_selector_event
                );
                return self.event_err(
                    event.correlation_id,
                    StatusCode::InvalidArgument as i32,
                    format!("CommitteeSelectorActor: unknown event {:?}", committee_selector_event),
                );
            }
        }
    }

    fn get_reference_values(&self) -> ReferenceValues {
        self.reference_values.clone()
    }
}

fn request_name(request: &CommitteeSelectorRequest) -> &'static str {
    match request.msg {
        Some(committee_selector_request::Msg::CreateCommittee(_)) => "CreateCommittee",
        Some(committee_selector_request::Msg::VolunteerBatchForCommittee(_)) => {
            "VolunteerBatchForCommittee"
        }
        Some(committee_selector_request::Msg::SampleCommittee(_)) => "SampleCommittee",
        Some(committee_selector_request::Msg::CheckCommitteeStatus(_)) => "CheckCommitteeStatus",
        _ => "Unknown",
    }
}

fn event_name(event: &CommitteeSelectorEvent) -> &'static str {
    match event.event {
        Some(committee_selector_event::Event::CreateCommitteeEvent(_)) => "CreateCommittee",
        Some(committee_selector_event::Event::VolunteerBatchForCommitteeEvent(_)) => {
            "VolunteerBatchForCommittee"
        }
        Some(committee_selector_event::Event::SampleCommitteeEvent(_)) => "SampleCommittee",
        _ => "Unknown",
    }
}
