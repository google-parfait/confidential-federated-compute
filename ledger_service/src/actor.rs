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

use alloc::{boxed::Box, vec};
use federated_compute::proto::{
    ledger_request::*, ledger_response::*, Ledger, LedgerConfig, LedgerRequest, LedgerResponse,
    Status,
};
use oak_crypto::signer::Signer;
use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    verifying_key_reference_value, AmdSevReferenceValues, ApplicationLayerReferenceValues,
    BinaryReferenceValue, ClaimReferenceValue, EndorsementReferenceValue, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, KeyType,
    OakRestrictedKernelReferenceValues, ReferenceValues, RootLayerReferenceValues,
    SkipVerification, StringLiterals, TcbVersion, TeePlatform, TextReferenceValue, VerifyingKey,
    VerifyingKeyReferenceValue, VerifyingKeySet,
};
use oak_restricted_kernel_sdk::Attester;
use prost::{bytes::Bytes, Message};
use slog::{debug, error, warn};
use tcp_runtime::model::{
    Actor, ActorCommand, ActorContext, ActorError, ActorEvent, ActorEventContext, CommandOutcome,
    EventOutcome,
};

pub struct LedgerActor {
    context: Option<Box<dyn ActorContext>>,
    ledger: LedgerService,
    platform_type: i32,
}

impl LedgerActor {
    pub fn create(attester: Box<dyn Attester>, signer: Box<dyn Signer>) -> anyhow::Result<Self> {
        let evidence = attester.quote()?;
        let platform_type = match evidence.root_layer {
            Some(root_layer) => root_layer.platform,
            None => TeePlatform::None.into(),
        };

        Ok(LedgerActor { context: None, ledger: LedgerService::new(signer), platform_type })
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

    fn get_reference_values(&self) -> ReferenceValues {
        let skip = BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
        };
        // When running in insecure mode, simply skip all reference values.
        // This is only used for tests.
        if self.platform_type == TeePlatform::None as i32 {
            ReferenceValues {
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
            }
        } else {
            let rekor_reference_value = VerifyingKeyReferenceValue {
                r#type: Some(verifying_key_reference_value::Type::Verify(VerifyingKeySet {
                    keys: vec![VerifyingKey {
                        r#type: KeyType::EcdsaP256Sha256.into(),
                        key_id: 1,
                        raw: vec![
                            0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02,
                            0x01, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03,
                            0x42, 0x00, 0x04, 0xd8, 0x6d, 0x98, 0xfb, 0x6b, 0x5a, 0x6d, 0xd4, 0xd5,
                            0xe4, 0x17, 0x06, 0x88, 0x12, 0x31, 0xd1, 0xaf, 0x5f, 0x00, 0x5c, 0x2b,
                            0x90, 0x16, 0xe6, 0x2d, 0x21, 0xad, 0x92, 0xce, 0x0b, 0xde, 0xa5, 0xfa,
                            0xc9, 0x86, 0x34, 0xce, 0xe7, 0xc1, 0x9e, 0x10, 0xbc, 0x52, 0xbf, 0xe2,
                            0xcb, 0x9e, 0x46, 0x85, 0x63, 0xff, 0xf4, 0x0f, 0xdb, 0x63, 0x62, 0xe1,
                            0x0b, 0x7d, 0x0c, 0xf7, 0xe4, 0x58, 0xb7,
                        ],
                    }],
                })),
            };
            ReferenceValues {
                r#type: Some(reference_values::Type::OakRestrictedKernel(
                    OakRestrictedKernelReferenceValues {
                        root_layer: Some(RootLayerReferenceValues {
                            amd_sev: Some(AmdSevReferenceValues {
                                min_tcb_version: Some(TcbVersion {
                                    boot_loader: 3,
                                    tee: 0,
                                    snp: 20,
                                    microcode: 209,
                                }),
                                stage0: Some(BinaryReferenceValue {
                                    r#type: Some(binary_reference_value::Type::Endorsement(
                                        EndorsementReferenceValue {
                                            endorser: Some(VerifyingKeySet {
                                                keys: vec![VerifyingKey {
                                                    r#type: KeyType::EcdsaP256Sha256.into(),
                                                    key_id: 1,
                                                    raw: vec![
                                                        0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a,
                                                        0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06,
                                                        0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03,
                                                        0x01, 0x07, 0x03, 0x42, 0x00, 0x04, 0xd3,
                                                        0x03, 0x16, 0xa0, 0x7b, 0xb4, 0xae, 0x53,
                                                        0x43, 0x6e, 0x51, 0x77, 0x23, 0x9f, 0xaf,
                                                        0x6e, 0x75, 0xb4, 0x5c, 0xd6, 0xd1, 0x2f,
                                                        0xdd, 0xe7, 0x4e, 0x76, 0xf6, 0xd2, 0xe7,
                                                        0xab, 0x4e, 0x54, 0x3b, 0x6e, 0x84, 0xdb,
                                                        0xf5, 0xc1, 0xf1, 0x9f, 0x2a, 0x36, 0x30,
                                                        0xbe, 0x0b, 0x77, 0x69, 0x87, 0x3e, 0x71,
                                                        0xdb, 0xda, 0xd0, 0xe6, 0x30, 0x18, 0x12,
                                                        0x88, 0x29, 0xfd, 0x97, 0x18, 0x4e, 0x07,
                                                    ],
                                                }],
                                            }),
                                            required_claims: Some(ClaimReferenceValue {
                                                claim_types: vec![
                                                    "https://github.com/project-oak/oak/blob/main/docs/tr/claim/66738.md".into(),
                                                    "https://github.com/project-oak/oak/blob/main/docs/tr/claim/10271.md".into(),
                                                ],
                                            }),
                                            rekor: Some(rekor_reference_value.clone()),
                                            ..Default::default()
                                        },
                                    )),
                                }),
                                ..Default::default()
                            }),
                            ..Default::default()
                        }),
                        kernel_layer: Some(KernelLayerReferenceValues {
                            kernel: Some(KernelBinaryReferenceValue {
                                r#type: Some(kernel_binary_reference_value::Type::Endorsement(
                                    EndorsementReferenceValue {
                                        endorser: Some(VerifyingKeySet {
                                            keys: vec![VerifyingKey {
                                                r#type: KeyType::EcdsaP256Sha256.into(),
                                                key_id: 1,
                                                raw: vec![
                                                    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86,
                                                    0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
                                                    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03,
                                                    0x42, 0x00, 0x04, 0x2b, 0x3a, 0x56, 0x7f, 0x3f,
                                                    0x0d, 0xa3, 0x4e, 0x44, 0x6c, 0x41, 0xf3, 0x87,
                                                    0x4c, 0x95, 0x5e, 0x2e, 0x1b, 0xa2, 0xb8, 0x86,
                                                    0xdd, 0x9b, 0x8e, 0xa2, 0xba, 0xa9, 0xd6, 0x00,
                                                    0xd9, 0xda, 0x1b, 0x35, 0xe7, 0x41, 0x86, 0x5c,
                                                    0x65, 0xb4, 0x0b, 0xd6, 0xdd, 0x19, 0x6d, 0xcd,
                                                    0x85, 0x45, 0x1e, 0x0d, 0xc1, 0x30, 0xa3, 0x49,
                                                    0xfe, 0x60, 0xd7, 0xbe, 0x73, 0x70, 0x5d, 0xb8,
                                                    0x23, 0xd1, 0xb1,
                                                ],
                                            }],
                                        }),
                                        required_claims: Some(ClaimReferenceValue {
                                            claim_types: vec![
                                                "https://github.com/project-oak/oak/blob/main/docs/tr/claim/36746.md".into(),
                                                "https://github.com/project-oak/oak/blob/main/docs/tr/claim/98982.md".into(),
                                            ],
                                        }),
                                        rekor: Some(rekor_reference_value.clone()),
                                        ..Default::default()
                                    },
                                )),
                            }),
                            kernel_cmd_line_text: Some(TextReferenceValue {
                                r#type: Some(text_reference_value::Type::StringLiterals(
                                    StringLiterals { value: vec!["console=ttyS0".into()] },
                                )),
                            }),
                            init_ram_fs: Some(BinaryReferenceValue {
                                r#type: Some(binary_reference_value::Type::Endorsement(
                                    EndorsementReferenceValue {
                                        endorser: Some(VerifyingKeySet {
                                            keys: vec![VerifyingKey {
                                                r#type: KeyType::EcdsaP256Sha256.into(),
                                                key_id: 1,
                                                raw: vec![
                                                    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86,
                                                    0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
                                                    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03,
                                                    0x42, 0x00, 0x04, 0x4e, 0x30, 0x6d, 0x9d, 0x7a,
                                                    0x06, 0xf7, 0x21, 0x48, 0x30, 0x5a, 0xf7, 0xcb,
                                                    0x77, 0x92, 0xf0, 0x56, 0x7b, 0x39, 0xf2, 0x01,
                                                    0x7f, 0x49, 0x43, 0x8e, 0xef, 0x67, 0xb0, 0xd6,
                                                    0xdb, 0x79, 0xaf, 0xce, 0x67, 0x9f, 0xf2, 0x0b,
                                                    0x6c, 0x04, 0xba, 0x63, 0xef, 0xb7, 0xd1, 0x68,
                                                    0xc4, 0x2e, 0x30, 0x8f, 0x05, 0xa1, 0x36, 0x77,
                                                    0x40, 0x69, 0x4c, 0x85, 0x5b, 0xf6, 0xbd, 0xf4,
                                                    0xe5, 0x7d, 0xd9,
                                                ],
                                            }],
                                        }),
                                        required_claims: Some(ClaimReferenceValue {
                                            claim_types: vec![
                                                "https://github.com/project-oak/oak/blob/main/docs/tr/claim/87425.md".into(),
                                            ],
                                        }),
                                        rekor: Some(rekor_reference_value.clone()),
                                        ..Default::default()
                                    },
                                )),
                            }),
                            memory_map: Some(skip.clone()),
                            acpi: Some(skip.clone()),
                            ..Default::default()
                        }),
                        application_layer: Some(ApplicationLayerReferenceValues {
                            binary: Some(BinaryReferenceValue {
                                r#type: Some(binary_reference_value::Type::Endorsement(
                                    EndorsementReferenceValue {
                                        endorser: Some(VerifyingKeySet {
                                            keys: vec![VerifyingKey {
                                                r#type: KeyType::EcdsaP256Sha256.into(),
                                                key_id: 8,
                                                raw: vec![
                                                    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86,
                                                    0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
                                                    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03,
                                                    0x42, 0x00, 0x04, 0x67, 0xcc, 0xec, 0x5a, 0x09,
                                                    0xdc, 0x93, 0x9c, 0x0b, 0x77, 0x24, 0x82, 0xdc,
                                                    0x01, 0xd9, 0x54, 0x67, 0xc6, 0x59, 0xac, 0x61,
                                                    0x90, 0x96, 0x2a, 0x9a, 0xa6, 0xea, 0xd6, 0x9a,
                                                    0xad, 0xb0, 0x9c, 0x12, 0x6d, 0xee, 0x65, 0x26,
                                                    0x77, 0x52, 0xd3, 0x3c, 0x68, 0x14, 0xfa, 0xa6,
                                                    0x32, 0x9b, 0xee, 0x88, 0xc2, 0xee, 0x62, 0x06,
                                                    0xc7, 0xa7, 0x94, 0xfd, 0x7b, 0x1f, 0x71, 0x39,
                                                    0x8c, 0x86, 0xf4,
                                                ],
                                            }],
                                        }),
                                        required_claims: Some(ClaimReferenceValue::default()),
                                        rekor: Some(rekor_reference_value.clone()),
                                        ..Default::default()
                                    },
                                )),
                            }),
                            configuration: Some(skip.clone()),
                        }),
                    },
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use oak_proto_rust::oak::attestation::v1::{
        Endorsements, Evidence, RootLayerEvidence, TeePlatform,
    };
    use oak_restricted_kernel_sdk::testing::MockSigner;
    use oak_restricted_kernel_sdk::Attester;
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

    struct MockAttester {
        platform: TeePlatform,
    }

    impl Attester for MockAttester {
        fn extend(&mut self, _encoded_event: &[u8]) -> anyhow::Result<()> {
            Ok(())
        }

        fn quote(&self) -> anyhow::Result<Evidence> {
            Ok(Evidence {
                root_layer: Some(RootLayerEvidence {
                    platform: self.platform.into(),
                    ..Default::default()
                }),
                ..Default::default()
            })
        }
    }

    fn create_actor() -> LedgerActor {
        create_actor_with_platform(TeePlatform::None)
    }

    fn create_actor_with_platform(platform: TeePlatform) -> LedgerActor {
        let mock_context = Box::new(MockActorContext::new());
        let mut actor = LedgerActor::create(
            Box::new(MockAttester { platform }),
            Box::new(MockSigner::create().unwrap()),
        )
        .unwrap();
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

    #[test]
    fn test_reference_values() -> anyhow::Result<()> {
        let actor = create_actor_with_platform(TeePlatform::AmdSevSnp);
        let reference_values = actor.get_reference_values();

        let evidence =
            Evidence::decode(include_bytes!("../testdata/evidence_20241121.binarypb").as_slice())?;
        let endorsements = Endorsements::decode(
            include_bytes!("../testdata/endorsements_20241121.binarypb").as_slice(),
        )?;
        assert_that!(
            oak_attestation_verification::verifier::verify(
                1732147200000,
                &evidence,
                &endorsements,
                &reference_values
            ),
            ok(anything())
        );
        Ok(())
    }
}
