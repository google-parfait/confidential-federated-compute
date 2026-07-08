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

use crate::decryptor::{convert_from_prost, convert_to_prost, ReputableDecryptorState};
use crate::secure_aggregation::willow::{
    reputable_decryptor_event, reputable_decryptor_request, reputable_decryptor_response,
    CreateSetupContributionRequest, CreateSetupContributionResponse, DecryptionEvent,
    HandlePartialDecryptionRequest, HandlePartialDecryptionResponse, ReputableDecryptorConfig,
    ReputableDecryptorEvent, ReputableDecryptorRequest, ReputableDecryptorResponse,
    ReputableDecryptorSnapshot, ReputableDecryptorStatus, SetupEvent,
    VerifyAndAggregateKeyContributionsRequest, VerifyAndAggregateKeyContributionsResponse,
};
use ahe_traits::AheBase;
use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use decryptor_traits::{SecureAggregationBaseMultiDecryptor, SecureAggregationReputableDecryptor};
use micro_rpc::StatusCode;
use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    ApplicationLayerReferenceValues, BinaryReferenceValue, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakRestrictedKernelReferenceValues,
    ReferenceValues, RootLayerReferenceValues, SkipVerification, TextReferenceValue,
};
use prng_traits::SecurePrng;
use prost::{bytes::Bytes, Message};
use proto_serialization_traits::{FromProto, ToProto};
use shell_kahe::ShellKahe;
use shell_vahe::ShellVahe;
use slog::{debug, warn};
use status::{self, StatusError};
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};
use willow_v1_decryptor::WillowV1Decryptor;

/// Replicated system actor wrapping the `ReputableDecryptor` for safe
/// deployment inside TCP enclaves.
pub struct ReputableDecryptorActor {
    reference_values: ReferenceValues,
    context: Option<Box<dyn ActorContext>>,
    max_number_of_decryptors: usize,
    state: ReputableDecryptorState,
}

impl ReputableDecryptorActor {
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
        ReputableDecryptorActor {
            reference_values,
            context: None,
            max_number_of_decryptors: 0,
            state: ReputableDecryptorState::default(),
        }
    }

    fn get_context(&mut self) -> &mut dyn ActorContext {
        self.context.as_mut().expect("Context is initialized").as_mut()
    }

    fn command_err(
        &mut self,
        correlation_id: u64,
        code: i32,
        message: String,
    ) -> Result<CommandOutcome, ActorError> {
        Ok(CommandOutcome::with_command(ActorCommand::with_header(
            correlation_id,
            &ReputableDecryptorResponse {
                msg: Some(reputable_decryptor_response::Msg::Error(ReputableDecryptorStatus {
                    code,
                    message,
                })),
            },
        )))
    }

    fn event_err(
        &mut self,
        correlation_id: u64,
        code: i32,
        message: String,
    ) -> Result<EventOutcome, ActorError> {
        Ok(EventOutcome::with_command(ActorCommand::with_header(
            correlation_id,
            &ReputableDecryptorResponse {
                msg: Some(reputable_decryptor_response::Msg::Error(ReputableDecryptorStatus {
                    code,
                    message,
                })),
            },
        )))
    }

    fn create_willow_multi_decryptor(
        &self,
        _key_id: Vec<u8>,
        seed: &[u8],
    ) -> WillowV1Decryptor<ShellVahe> {
        let config =
            shell_parameters::create_shell_ahe_config(self.max_number_of_decryptors as i64)
                .expect("failed to create Shell AHE config");
        let vahe = Rc::new(
            ShellVahe::new(config, b"willow_reputable_decryptor")
                .expect("failed to initialize ShellVahe"),
        );
        let prng_seed = shell_prng_seed_from_bytes(seed).unwrap();
        let prng = core::cell::RefCell::new(
            <<ShellVahe as ahe_traits::AheBase>::Rng as prng_traits::SecurePrng>::create(
                &prng_seed,
            )
            .expect("failed to create prng"),
        );
        WillowV1Decryptor { vahe, prng }
    }

    /// Helper to dynamically and deterministically generate a setup
    /// contribution (containing public key share and ZK relation proof) on
    /// the fly from a compact stored PRNG seed.
    ///
    /// This is used both to fulfill duplicate/idempotent client requests and by
    /// the leader replica when committing a setup event.
    fn generate_setup_contribution(
        &self,
        key_id: Vec<u8>,
        prng_seed: &[u8],
    ) -> Result<crate::secure_aggregation::willow::SetupContribution, StatusError> {
        let decryptor = self.create_willow_multi_decryptor(key_id, prng_seed);
        let mut decryptor_state = willow_v1_decryptor::DecryptorState::default();
        let setup_contrib = decryptor.create_setup_contribution(&mut decryptor_state)?;
        let pb_proto = setup_contrib.to_proto(&decryptor)?;
        convert_to_prost(&pb_proto)
    }

    fn process_create_setup_contribution_command(
        &mut self,
        correlation_id: u64,
        req: CreateSetupContributionRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let key_id = &req.key_id;
        if key_id.is_empty() {
            return self.command_err(
                correlation_id,
                StatusCode::InvalidArgument as i32,
                "missing key_id field".to_string(),
            );
        }

        // Return on-demand regenerated contribution if we already have the seed.
        if let Some(state) = self.state.decryptor_states.get(key_id) {
            match self.generate_setup_contribution(key_id.as_bytes().to_vec(), &state.prng_seed) {
                Ok(prost_contrib) => {
                    return Ok(CommandOutcome::with_command(ActorCommand::with_header(
                        correlation_id,
                        &ReputableDecryptorResponse {
                            msg: Some(reputable_decryptor_response::Msg::CreateSetupContribution(
                                CreateSetupContributionResponse {
                                    setup_contribution: Some(prost_contrib),
                                },
                            )),
                        },
                    )));
                }
                Err(e) => {
                    return self.command_err(
                        correlation_id,
                        StatusCode::Internal as i32,
                        format!("failed to regenerate setup contribution for {}: {:?}", key_id, e),
                    );
                }
            }
        }

        // Leader samples key generation randomness to keep replicas in sync.
        let prng_seed = rand::random::<[u8; 32]>().to_vec();

        let mut event = ReputableDecryptorEvent::default();
        let mut setup_event = SetupEvent::default();
        setup_event.key_id = key_id.clone();
        setup_event.prng_seed = prng_seed;
        event.event = Some(reputable_decryptor_event::Event::SetupEvent(setup_event));

        Ok(CommandOutcome::with_event(ActorEvent::with_proto(correlation_id, &event)))
    }

    fn process_create_setup_contribution_event(
        &mut self,
        context: ActorEventContext,
        correlation_id: u64,
        event: SetupEvent,
    ) -> Result<EventOutcome, ActorError> {
        let key_id = &event.key_id;

        // All consensus replicas persist ONLY the 32-byte seed in their state, avoiding
        // the overhead of replicating or storing large polynomial-based keys.
        let (key_state, _) = self.state.get_or_insert_key_state(key_id, event.prng_seed.clone());

        // Only the leader replica (context.owned is true) needs to generate the key
        // pair on the fly to return the response to the client.
        if !context.owned {
            return Ok(EventOutcome::with_none());
        }

        let seed = key_state.prng_seed.clone();
        let run = || -> Result<ReputableDecryptorResponse, (StatusCode, String)> {
            let prost_contrib = self
                .generate_setup_contribution(key_id.as_bytes().to_vec(), &seed)
                .map_err(|e| {
                    (
                        StatusCode::Internal,
                        format!(
                            "failed to convert setup contribution to prost for {}: {:?}",
                            key_id, e
                        ),
                    )
                })?;
            Ok(ReputableDecryptorResponse {
                msg: Some(reputable_decryptor_response::Msg::CreateSetupContribution(
                    CreateSetupContributionResponse { setup_contribution: Some(prost_contrib) },
                )),
            })
        };
        match run() {
            Ok(resp) => {
                Ok(EventOutcome::with_command(ActorCommand::with_header(correlation_id, &resp)))
            }
            Err((code, msg)) => self.event_err(correlation_id, code as i32, msg),
        }
    }

    fn process_handle_partial_decryption_request_command(
        &mut self,
        correlation_id: u64,
        req: HandlePartialDecryptionRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let key_id = &req.key_id;
        if key_id.is_empty() {
            return self.command_err(
                correlation_id,
                StatusCode::InvalidArgument as i32,
                "missing key_id field".to_string(),
            );
        }

        if !self.state.decryptor_states.contains_key(key_id) {
            return self.command_err(
                correlation_id,
                StatusCode::FailedPrecondition as i32,
                format!("key_id {} not found", key_id),
            );
        }

        let Some(partial_decryption_request) = req.partial_decryption_request else {
            return self.command_err(
                correlation_id,
                StatusCode::InvalidArgument as i32,
                "missing partial_decryption_request field".to_string(),
            );
        };

        let mut event = ReputableDecryptorEvent::default();
        let mut decryption_event = DecryptionEvent::default();
        decryption_event.key_id = key_id.clone();
        decryption_event.partial_decryption_request = Some(partial_decryption_request);
        event.event = Some(reputable_decryptor_event::Event::DecryptionEvent(decryption_event));

        Ok(CommandOutcome::with_event(ActorEvent::with_proto(correlation_id, &event)))
    }

    fn process_handle_partial_decryption_event(
        &mut self,
        context: ActorEventContext,
        correlation_id: u64,
        event: DecryptionEvent,
    ) -> Result<EventOutcome, ActorError> {
        let key_id = &event.key_id;

        let key_state = match self.state.decryptor_states.remove(key_id) {
            Some(s) => s,
            None => {
                if context.owned {
                    return self.event_err(
                        correlation_id,
                        StatusCode::FailedPrecondition as i32,
                        format!("key_id {} not found or already used for decryption", key_id),
                    );
                }
                return Ok(EventOutcome::with_none());
            }
        };

        if !context.owned {
            return Ok(EventOutcome::with_none());
        }

        let run = || -> Result<ReputableDecryptorResponse, (StatusCode, String)> {
            let req_msg = event.partial_decryption_request.as_ref().ok_or((
                StatusCode::InvalidArgument,
                "missing partial_decryption_request field".to_string(),
            ))?;
            let req_msg_pb: messages_rust_proto::PartialDecryptionRequest =
                convert_from_prost(req_msg).map_err(|e| {
                    (
                        StatusCode::InvalidArgument,
                        format!("failed to convert partial decryption request from prost: {:?}", e),
                    )
                })?;
            let key_id_bytes = key_id.as_bytes().to_vec();
            let decryptor = self.create_willow_multi_decryptor(key_id_bytes, &key_state.prng_seed);
            let high_level_req =
                messages::PartialDecryptionRequest::<ShellVahe>::from_proto(req_msg_pb, &decryptor)
                    .map_err(|e| {
                        (
                            StatusCode::InvalidArgument,
                            format!(
                                "failed to parse partial decryption request from proto: {:?}",
                                e
                            ),
                        )
                    })?;
            let mut decryptor_state = willow_v1_decryptor::DecryptorState::default();
            decryptor.create_setup_contribution(&mut decryptor_state).map_err(|e| {
                (StatusCode::Internal, format!("failed to regenerate setup contribution: {:?}", e))
            })?;
            let response = decryptor
                .handle_partial_decryption_request::<ShellKahe>(
                    high_level_req,
                    None,
                    &mut decryptor_state,
                )
                .map_err(|e| {
                    (
                        StatusCode::Internal,
                        format!("failed to handle partial decryption request: {:?}", e),
                    )
                })?;
            let pb_response = response.to_proto((&decryptor, None)).map_err(|e| {
                (StatusCode::Internal, format!("failed to convert response to proto: {:?}", e))
            })?;
            let prost_response = convert_to_prost(&pb_response).map_err(|e| {
                (StatusCode::Internal, format!("failed to convert response to prost: {:?}", e))
            })?;
            Ok(ReputableDecryptorResponse {
                msg: Some(reputable_decryptor_response::Msg::HandlePartialDecryption(
                    HandlePartialDecryptionResponse {
                        partial_decryption_response: Some(prost_response),
                    },
                )),
            })
        };
        match run() {
            Ok(resp) => {
                Ok(EventOutcome::with_command(ActorCommand::with_header(correlation_id, &resp)))
            }
            Err((code, msg)) => self.event_err(correlation_id, code as i32, msg),
        }
    }

    fn process_verify_and_aggregate_key_contributions_command(
        &mut self,
        correlation_id: u64,
        req: VerifyAndAggregateKeyContributionsRequest,
    ) -> Result<CommandOutcome, ActorError> {
        let key_id = &req.key_id;
        if key_id.is_empty() {
            return self.command_err(
                correlation_id,
                StatusCode::InvalidArgument as i32,
                "missing key_id field".to_string(),
            );
        }

        let run = || -> Result<ReputableDecryptorResponse, (StatusCode, String)> {
            let inner_req = req.verify_key_contributions_request.as_ref().ok_or((
                StatusCode::InvalidArgument,
                "missing verify_key_contributions_request field".to_string(),
            ))?;
            let state = self
                .state
                .decryptor_states
                .get(key_id)
                .ok_or((StatusCode::FailedPrecondition, format!("key_id {} not found", key_id)))?;
            let key_id_bytes = key_id.as_bytes().to_vec();
            let decryptor = self.create_willow_multi_decryptor(key_id_bytes, &state.prng_seed);
            let inner_req_pb: messages_rust_proto::VerifyKeyContributionsRequest =
                convert_from_prost(inner_req).map_err(|e| {
                    (
                        StatusCode::InvalidArgument,
                        format!(
                            "failed to convert verify key contributions request from prost: {:?}",
                            e
                        ),
                    )
                })?;
            let high_level_req = messages::VerifyKeyContributionsRequest::<ShellVahe>::from_proto(
                inner_req_pb,
                &decryptor,
            )
            .map_err(|e| {
                (
                    StatusCode::InvalidArgument,
                    format!("failed to parse verify key contributions request from proto: {:?}", e),
                )
            })?;
            let aggregated_pk =
                decryptor.verify_and_aggregate_key_contributions(high_level_req).map_err(|e| {
                    let status_code = match e.code() {
                        status::StatusErrorCode::INVALID_ARGUMENT => StatusCode::InvalidArgument,
                        status::StatusErrorCode::FAILED_PRECONDITION => {
                            StatusCode::FailedPrecondition
                        }
                        _ => StatusCode::Internal,
                    };
                    (
                        status_code,
                        format!("failed to verify and aggregate key contributions: {:?}", e),
                    )
                })?;
            let pb_pk = aggregated_pk.to_proto(decryptor.vahe.as_ref()).map_err(|e| {
                (StatusCode::Internal, format!("failed to convert public key to proto: {:?}", e))
            })?;
            let prost_pk = convert_to_prost(&pb_pk).map_err(|e| {
                (StatusCode::Internal, format!("failed to convert public key to prost: {:?}", e))
            })?;
            Ok(ReputableDecryptorResponse {
                msg: Some(reputable_decryptor_response::Msg::VerifyAndAggregateKeyContributions(
                    VerifyAndAggregateKeyContributionsResponse { public_key: Some(prost_pk) },
                )),
            })
        };

        match run() {
            Ok(resp) => {
                Ok(CommandOutcome::with_command(ActorCommand::with_header(correlation_id, &resp)))
            }
            Err((code, msg)) => self.command_err(correlation_id, code as i32, msg),
        }
    }
}

impl Actor for ReputableDecryptorActor {
    fn on_init(&mut self, context: Box<dyn ActorContext>) -> Result<(), ActorError> {
        self.context = Some(context);
        let config = ReputableDecryptorConfig::decode(self.get_context().config().as_ref())
            .map_err(|_| ActorError::ConfigLoading)?;
        self.max_number_of_decryptors = config.max_number_of_decryptors as usize;
        self.state.max_number_of_decryptor_states = config.max_number_of_keys as usize;
        Ok(())
    }

    fn on_shutdown(&mut self) {}

    fn on_save_snapshot(&mut self) -> Result<Bytes, ActorError> {
        let snapshot = self.state.save_snapshot().map_err(|_| ActorError::Internal)?;
        Ok(snapshot.encode_to_vec().into())
    }

    fn on_load_snapshot(&mut self, snapshot_bytes: Bytes) -> Result<(), ActorError> {
        debug!(self.get_context().logger(), "Loading snapshot");

        let snapshot = ReputableDecryptorSnapshot::decode(snapshot_bytes)
            .map_err(|_| ActorError::SnapshotLoading)?;

        self.state.load_snapshot(snapshot).map_err(|_| ActorError::SnapshotLoading)?;

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

        let request = ReputableDecryptorRequest::decode(command.header).map_err(|error| {
            warn!(
                self.get_context().logger(),
                "ReputableDecryptorActor: request cannot be parsed: {}", error
            );
            ActorError::Internal
        })?;

        debug!(
            self.get_context().logger(),
            "ReputableDecryptorActor: handling {} command",
            request_name(&request)
        );

        if !self.get_context().leader() {
            warn!(
                self.get_context().logger(),
                "ReputableDecryptorActor: command {} rejected: not a leader",
                request_name(&request)
            );
            return self.command_err(
                command.correlation_id,
                StatusCode::FailedPrecondition as i32,
                "Node is not the leader".to_string(),
            );
        }

        match request.msg {
            Some(reputable_decryptor_request::Msg::CreateSetupContribution(req)) => {
                self.process_create_setup_contribution_command(command.correlation_id, req)
            }

            Some(reputable_decryptor_request::Msg::HandlePartialDecryption(req)) => {
                self.process_handle_partial_decryption_request_command(command.correlation_id, req)
            }
            Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(req)) => self
                .process_verify_and_aggregate_key_contributions_command(
                    command.correlation_id,
                    req,
                ),
            _ => {
                warn!(
                    self.get_context().logger(),
                    "ReputableDecryptorActor: unknown request {:?}", request
                );
                self.command_err(
                    command.correlation_id,
                    StatusCode::InvalidArgument as i32,
                    format!("ReputableDecryptorActor: unknown request {:?}", request),
                )
            }
        }
    }

    fn on_apply_event(
        &mut self,
        context: ActorEventContext,
        event: ActorEvent,
    ) -> Result<EventOutcome, ActorError> {
        let dec_event = ReputableDecryptorEvent::decode(event.contents).map_err(|error| {
            warn!(
                self.get_context().logger(),
                "ReputableDecryptorActor: event cannot be parsed: {}", error
            );
            ActorError::Internal
        })?;

        debug!(
            self.get_context().logger(),
            "ReputableDecryptorActor: applying {} event",
            event_name(&dec_event)
        );

        match dec_event.event {
            Some(reputable_decryptor_event::Event::SetupEvent(e)) => {
                self.process_create_setup_contribution_event(context, event.correlation_id, e)
            }
            Some(reputable_decryptor_event::Event::DecryptionEvent(e)) => {
                self.process_handle_partial_decryption_event(context, event.correlation_id, e)
            }

            _ => {
                warn!(
                    self.get_context().logger(),
                    "ReputableDecryptorActor: unknown event {:?}", dec_event
                );
                self.event_err(
                    event.correlation_id,
                    StatusCode::InvalidArgument as i32,
                    format!("ReputableDecryptorActor: unknown event {:?}", dec_event),
                )
            }
        }
    }

    fn get_reference_values(&self) -> ReferenceValues {
        self.reference_values.clone()
    }
}

fn request_name(request: &ReputableDecryptorRequest) -> &'static str {
    match request.msg {
        Some(reputable_decryptor_request::Msg::CreateSetupContribution(_)) => {
            "CreateSetupContribution"
        }

        Some(reputable_decryptor_request::Msg::HandlePartialDecryption(_)) => {
            "HandlePartialDecryption"
        }
        Some(reputable_decryptor_request::Msg::VerifyAndAggregateKeyContributions(_)) => {
            "VerifyAndAggregateKeyContributions"
        }
        _ => "Unknown",
    }
}

fn event_name(event: &ReputableDecryptorEvent) -> &'static str {
    match event.event {
        Some(reputable_decryptor_event::Event::SetupEvent(_)) => "SetupEvent",
        Some(reputable_decryptor_event::Event::DecryptionEvent(_)) => "DecryptionEvent",

        _ => "Unknown",
    }
}

fn shell_prng_seed_from_bytes(
    bytes: &[u8],
) -> Result<<<ShellVahe as AheBase>::Rng as SecurePrng>::Seed, StatusError> {
    single_thread_hkdf::compute_hkdf(bytes, &[], &[], single_thread_hkdf::seed_length())
}
