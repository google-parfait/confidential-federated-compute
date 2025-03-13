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

use googletest::prelude::*;
use kms_proto::fcp::confidentialcompute::SessionResponseWithStatus;
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_proto_rust::oak::{
    attestation::v1::ReferenceValues,
    session::v1::{PlaintextMessage, SessionRequestWithSessionId, SessionResponse},
};
use oak_session::{ClientSession, ProtocolEngine, Session};
use prost::{bytes::Bytes, Message};
use rand::Rng;
use session_config::create_session_config;
use session_test_utils::{
    test_reference_values, FakeAttester, FakeClock, FakeEndorser, FakeSigner,
};
use slog::Drain;
use storage_actor::StorageActor;
use storage_proto::{
    confidential_federated_compute::kms::{
        read_request, read_response, storage_request, storage_response, update_request,
        ReadRequest, ReadResponse, StorageRequest, StorageResponse, UpdateRequest, UpdateResponse,
    },
    duration_proto::google::protobuf::Duration,
    status_proto::google::rpc::Status,
    timestamp_proto::google::protobuf::Timestamp,
};
use tcp_runtime::{
    mock::MockActorContext,
    model::{Actor, ActorCommand, ActorEvent, ActorEventContext, CommandOutcome},
};
use tonic::Code;

/// Creates a fake ActorContext for a leader node.
fn create_actor_context(leader: bool) -> Box<MockActorContext> {
    let mut mock_context = MockActorContext::new();
    mock_context
        .expect_logger()
        .return_const(slog::Logger::root(slog_stdlog::StdLog {}.fuse(), slog::o!()));
    mock_context.expect_id().return_const(0u64);
    mock_context.expect_config().return_const(Bytes::default());
    mock_context.expect_leader().return_const(leader);
    Box::new(mock_context)
}

/// Initializes a ClientSession with the actor.
fn create_client_session(
    actor: &mut StorageActor,
    session_id: &[u8],
    attester: Arc<FakeAttester>,
    endorser: Arc<FakeEndorser>,
    signer: FakeSigner,
    reference_values: ReferenceValues,
    clock: Arc<FakeClock>,
) -> ClientSession {
    let mut session = create_session_config(
        &(attester as Arc<dyn Attester>),
        &(endorser as Arc<dyn Endorser>),
        Box::new(signer),
        reference_values,
        clock,
    )
    .and_then(ClientSession::create)
    .expect("failed to create ClientSession");

    // Perform the initial handshake.
    let mut rng = rand::thread_rng();
    while !session.is_open() {
        let message = session.get_outgoing_message();
        assert_that!(message, ok(some(anything())));

        let correlation_id: u64 = rng.gen();
        let outcome = actor.on_process_command(Some(ActorCommand::with_header(
            correlation_id,
            &SessionRequestWithSessionId {
                session_id: session_id.into(),
                request: Some(message.unwrap().unwrap()),
            },
        )));
        assert_that!(
            outcome,
            ok(matches_pattern!(CommandOutcome {
                commands: elements_are![matches_pattern!(ActorCommand {
                    correlation_id: eq(correlation_id),
                })],
                event: none(),
            }))
        );

        let response =
            SessionResponseWithStatus::decode(outcome.unwrap().commands.pop().unwrap().header);
        assert_that!(response, ok(matches_pattern!(SessionResponseWithStatus { status: none() })));
        if let Some(response) = response.unwrap().response {
            assert_that!(
                session.put_incoming_message(
                    &SessionResponse::decode(response.encode_to_vec().as_slice())
                        .expect("failed to convert SessionResponse")
                ),
                ok(anything())
            );
        }
    }

    session
}

/// Uses the `ClientSession` to encrypt a request.
fn encode_request(
    session: &mut ClientSession,
    session_id: &[u8],
    request: &StorageRequest,
) -> SessionRequestWithSessionId {
    assert_that!(
        session.write(PlaintextMessage { plaintext: request.encode_to_vec() }),
        ok(anything())
    );
    let message = session.get_outgoing_message();
    assert_that!(message, ok(some(anything())));

    SessionRequestWithSessionId {
        session_id: session_id.into(),
        request: Some(message.unwrap().unwrap()),
    }
}

/// Uses the `ClientSession` to decrypt a response.
fn decode_response(
    session: &mut ClientSession,
    command: &ActorCommand,
) -> std::result::Result<StorageResponse, Status> {
    let response = SessionResponseWithStatus::decode(command.header.clone());
    assert_that!(response, ok(anything()));
    let status = response.as_ref().unwrap().status.clone().unwrap_or_default();
    if status.code != 0 {
        return Err(status);
    }
    assert_that!(
        response,
        ok(matches_pattern!(SessionResponseWithStatus { response: some(anything()) }))
    );
    assert_that!(
        session.put_incoming_message(
            &SessionResponse::decode(
                response.unwrap().response.unwrap().encode_to_vec().as_slice()
            )
            .expect("failed to convert SessionResponse")
        ),
        ok(anything())
    );
    let response = session.read();
    assert_that!(response, ok(some(anything())));
    let response = StorageResponse::decode(response.unwrap().unwrap().plaintext.as_slice());
    assert_that!(response, ok(anything()));
    Ok(response.unwrap())
}

#[test_log::test(googletest::test)]
fn get_reference_values_succeeds() {
    let reference_values = test_reference_values();
    let actor = StorageActor::new(
        Arc::new(FakeAttester::create().unwrap()),
        Arc::new(FakeEndorser::default()),
        FakeSigner::create().unwrap(),
        reference_values.clone(),
        Arc::new(FakeClock { milliseconds_since_epoch: 0 }),
    );

    expect_that!(actor.get_reference_values(), eq(reference_values));
}

#[test_log::test(googletest::test)]
fn empty_command_ignored_on_follower() {
    let mut actor = StorageActor::new(
        Arc::new(FakeAttester::create().unwrap()),
        Arc::new(FakeEndorser::default()),
        FakeSigner::create().unwrap(),
        test_reference_values(),
        Arc::new(FakeClock { milliseconds_since_epoch: 0 }),
    );
    assert_that!(actor.on_init(create_actor_context(/* leader= */ false)), ok(anything()));

    expect_that!(actor.on_process_command(None), ok(eq(CommandOutcome::with_none())));
}

#[test_log::test(googletest::test)]
fn empty_command_causes_periodic_clock_update() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 12_345_000 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    let mut seq = mockall::Sequence::new();
    let mut context = create_actor_context(true);
    context.expect_instant().times(1).in_sequence(&mut seq).return_const(100_000u64);
    context.expect_instant().times(1).in_sequence(&mut seq).return_const(150_000u64);
    context.expect_instant().times(1).in_sequence(&mut seq).return_const(160_000u64);
    assert_that!(actor.on_init(context), ok(anything()));

    let session_id = b"session-id";
    let mut session = create_client_session(
        &mut actor,
        session_id,
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );

    // The time should start at 0.
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 123,
                kind: Some(storage_request::Kind::Read(ReadRequest::default())),
            },
        ),
    )));
    assert_that!(outcome, ok(matches_pattern!(CommandOutcome { commands: len(eq(1)) })));
    assert_that!(
        decode_response(&mut session, &outcome.unwrap().commands[0]),
        ok(matches_pattern!(StorageResponse {
            kind: some(matches_pattern!(storage_response::Kind::Read(matches_pattern!(
                ReadResponse { now: some(matches_pattern!(Timestamp { seconds: eq(0) })) }
            )))),
        }))
    );

    // When the first empty command is sent, an event to update the time should
    // be generated since it's been more than a minute since the start time (0).
    let outcome = actor.on_process_command(None);
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome { commands: elements_are![], event: some(anything()) }))
    );
    // When the event is applied, no response should be generated but the time
    // should be updated.
    let outcome = actor
        .on_apply_event(
            ActorEventContext { owned: true, ..Default::default() },
            outcome.unwrap().event.unwrap(),
        )
        .expect("on_apply_event failed");
    expect_that!(outcome.commands, elements_are![]);

    // The time should now be 12345 seconds.
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        2,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 456,
                kind: Some(storage_request::Kind::Read(ReadRequest::default())),
            },
        ),
    )));
    assert_that!(outcome, ok(matches_pattern!(CommandOutcome { commands: len(eq(1)) })));
    expect_that!(
        decode_response(&mut session, &outcome.unwrap().commands[0]),
        ok(matches_pattern!(StorageResponse {
            kind: some(matches_pattern!(storage_response::Kind::Read(matches_pattern!(
                ReadResponse { now: some(matches_pattern!(Timestamp { seconds: eq(12345) })) }
            )))),
        }))
    );

    // On the next call, time will have advanced 50 seconds. A new event should
    // not be generated.
    expect_that!(actor.on_process_command(None), ok(eq(CommandOutcome::with_none())));

    // On the third call, time will have advanced 10 seconds. Since 60 seconds
    // have elapsed since the last event, a new one should be generated.
    expect_that!(
        actor.on_process_command(None),
        ok(matches_pattern!(CommandOutcome { commands: elements_are![], event: some(anything()) }))
    );
}

#[test_log::test(googletest::test)]
fn invalid_request_fails() {
    let mut actor = StorageActor::new(
        Arc::new(FakeAttester::create().unwrap()),
        Arc::new(FakeEndorser::default()),
        FakeSigner::create().unwrap(),
        test_reference_values(),
        Arc::new(FakeClock { milliseconds_since_epoch: 0 }),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let outcome = actor.on_process_command(Some(ActorCommand {
        correlation_id: 1,
        header: Bytes::from_static(b"invalid"),
        ..Default::default()
    }));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })],
            event: none(),
        }))
    );
    expect_that!(
        SessionResponseWithStatus::decode(outcome.unwrap().commands[0].header.clone()),
        ok(matches_pattern!(SessionResponseWithStatus {
            status: some(matches_pattern!(Status {
                code: eq(Code::InvalidArgument as i32),
                message: contains_substring("failed to decode SessionRequest"),
            }))
        }))
    );
}

#[test_log::test(googletest::test)]
fn request_to_follower_fails() {
    let mut actor = StorageActor::new(
        Arc::new(FakeAttester::create().unwrap()),
        Arc::new(FakeEndorser::default()),
        FakeSigner::create().unwrap(),
        test_reference_values(),
        Arc::new(FakeClock { milliseconds_since_epoch: 0 }),
    );
    assert_that!(actor.on_init(create_actor_context(/* leader= */ false)), ok(anything()));

    let outcome = actor.on_process_command(Some(ActorCommand {
        correlation_id: 1,
        header: SessionRequestWithSessionId::default().encode_to_vec().into(),
        ..Default::default()
    }));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })],
            event: none(),
        }))
    );
    expect_that!(
        SessionResponseWithStatus::decode(outcome.unwrap().commands[0].header.clone()),
        ok(matches_pattern!(SessionResponseWithStatus {
            status: some(matches_pattern!(Status {
                code: eq(Code::Aborted as i32),
                message: contains_substring("not a leader"),
            }))
        }))
    );
}

#[test_log::test(googletest::test)]
fn empty_storage_request_fails() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 0 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let session_id = b"session-id";
    let mut session = create_client_session(
        &mut actor,
        session_id,
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );

    // Run an invalid command.
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &encode_request(&mut session, session_id, &StorageRequest::default()),
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })],
            event: none(),
        }))
    );
    assert_that!(
        decode_response(&mut session, &outcome.unwrap().commands[0]),
        err(matches_pattern!(Status {
            code: eq(Code::InvalidArgument as i32),
            message: contains_substring("unsupported StorageRequest.kind"),
        }))
    );
}

#[test_log::test(googletest::test)]
fn write_and_read_succeeds() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 100_000 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let session_id = b"session-id";
    let mut session = create_client_session(
        &mut actor,
        session_id,
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );

    // Run an update command (5 -> "value").
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 123,
                kind: Some(storage_request::Kind::Update(UpdateRequest {
                    updates: vec![update_request::Update {
                        key: 5u128.to_be_bytes().into(),
                        value: Some(b"value".into()),
                        ..Default::default()
                    }],
                })),
            },
        ),
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![],
            event: some(matches_pattern!(ActorEvent { correlation_id: eq(1) })),
        }))
    );
    let outcome = actor
        .on_apply_event(
            ActorEventContext { owned: true, ..Default::default() },
            outcome.unwrap().event.unwrap(),
        )
        .expect("on_apply_event failed");
    assert_that!(
        outcome.commands,
        elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })]
    );
    assert_that!(
        decode_response(&mut session, &outcome.commands[0]),
        ok(matches_pattern!(StorageResponse {
            correlation_id: eq(123),
            kind: some(matches_pattern!(storage_response::Kind::Update(eq(
                UpdateResponse::default()
            )))),
        }))
    );

    // Run a read command for key 5.
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        2,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 456,
                kind: Some(storage_request::Kind::Read(ReadRequest {
                    ranges: vec![read_request::Range {
                        start: 5u128.to_be_bytes().into(),
                        ..Default::default()
                    }],
                })),
            },
        ),
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(2) })],
            event: none(),
        }))
    );
    assert_that!(
        decode_response(&mut session, &outcome.unwrap().commands[0]),
        ok(matches_pattern!(StorageResponse {
            correlation_id: eq(456),
            kind: some(matches_pattern!(storage_response::Kind::Read(matches_pattern!(
                ReadResponse {
                    now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
                    entries: elements_are![matches_pattern!(read_response::Entry {
                        key: eq(5u128.to_be_bytes()),
                        value: eq(b"value"),
                    })]
                }
            )))),
        }))
    );
}

#[test_log::test(googletest::test)]
fn save_and_load_snapshot_succeeds() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 100_000 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let session_id = b"session-id";
    let mut session = create_client_session(
        &mut actor,
        session_id,
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );

    // Run an update command (5 -> "value").
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 123,
                kind: Some(storage_request::Kind::Update(UpdateRequest {
                    updates: vec![update_request::Update {
                        key: 5u128.to_be_bytes().into(),
                        value: Some(b"value".into()),
                        ttl: Some(Duration { seconds: 1000, ..Default::default() }),
                        ..Default::default()
                    }],
                })),
            },
        ),
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![],
            event: some(matches_pattern!(ActorEvent { correlation_id: eq(1) })),
        }))
    );
    let outcome = actor
        .on_apply_event(
            ActorEventContext { owned: true, ..Default::default() },
            outcome.unwrap().event.unwrap(),
        )
        .expect("on_apply_event failed");
    assert_that!(
        outcome.commands,
        elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })]
    );
    assert_that!(
        decode_response(&mut session, &outcome.commands[0]),
        ok(matches_pattern!(StorageResponse { correlation_id: eq(123) }))
    );

    // Save a snapshot.
    let snapshot = actor.on_save_snapshot();
    assert_that!(snapshot, ok(anything()));

    // Load the snapshot on a new actor.
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        // This clock shouldn't affect the loaded snapshot.
        Arc::new(FakeClock { milliseconds_since_epoch: 200_000 }),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));
    assert_that!(actor.on_load_snapshot(snapshot.unwrap()), ok(anything()));

    // Run a read command for key 5.
    let mut session = create_client_session(
        &mut actor,
        session_id,
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        2,
        &encode_request(
            &mut session,
            session_id,
            &StorageRequest {
                correlation_id: 456,
                kind: Some(storage_request::Kind::Read(ReadRequest {
                    ranges: vec![read_request::Range {
                        start: 5u128.to_be_bytes().into(),
                        ..Default::default()
                    }],
                })),
            },
        ),
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(2) })],
            event: none(),
        }))
    );
    assert_that!(
        decode_response(&mut session, &outcome.unwrap().commands[0]),
        ok(matches_pattern!(StorageResponse {
            correlation_id: eq(456),
            kind: some(matches_pattern!(storage_response::Kind::Read(matches_pattern!(
                ReadResponse {
                    now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
                    entries: elements_are![matches_pattern!(read_response::Entry {
                        key: eq(5u128.to_be_bytes()),
                        value: eq(b"value"),
                        expiration: some(matches_pattern!(Timestamp { seconds: eq(1100) })),
                    })]
                }
            )))),
        }))
    );
}

#[test_log::test(googletest::test)]
fn session_reuse_fails() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 0 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let session_id = b"session-id";
    create_client_session(
        &mut actor,
        session_id,
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );

    // Attempting to create another session with the same ID should fail during the
    // handshake.
    let mut session = create_session_config(
        &(attester as Arc<dyn Attester>),
        &(endorser as Arc<dyn Endorser>),
        Box::new(signer),
        reference_values,
        clock,
    )
    .and_then(ClientSession::create)
    .expect("failed to create ClientSession");
    let message = session.get_outgoing_message();
    assert_that!(message, ok(some(anything())));
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &SessionRequestWithSessionId {
            session_id: session_id.into(),
            request: Some(message.unwrap().unwrap()),
        },
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })],
            event: none(),
        }))
    );
    assert_that!(
        SessionResponseWithStatus::decode(outcome.unwrap().commands[0].header.clone()),
        ok(matches_pattern!(SessionResponseWithStatus {
            status: some(matches_pattern!(Status {
                code: eq(Code::InvalidArgument as i32),
                message: contains_substring("SessionRequest is invalid"),
            }))
        }))
    );
}

#[test_log::test(googletest::test)]
fn session_reuse_after_close_succeeds() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 0 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    let session_id = b"session-id";
    create_client_session(
        &mut actor,
        session_id,
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );

    // Sending a message without a SessionRequest should close the session.
    let outcome = actor.on_process_command(Some(ActorCommand::with_header(
        1,
        &SessionRequestWithSessionId { session_id: session_id.into(), request: None },
    )));
    assert_that!(
        outcome,
        ok(matches_pattern!(CommandOutcome {
            commands: elements_are![matches_pattern!(ActorCommand { correlation_id: eq(1) })],
            event: none(),
        }))
    );
    assert_that!(
        SessionResponseWithStatus::decode(outcome.unwrap().commands[0].header.clone()),
        ok(matches_pattern!(SessionResponseWithStatus { status: none() }))
    );

    // Once the session is closed, a new session with the same ID should succeed.
    create_client_session(
        &mut actor,
        session_id,
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );
}

#[test_log::test(googletest::test)]
fn multiple_sessions_succeeds() {
    let attester = Arc::new(FakeAttester::create().unwrap());
    let endorser = Arc::new(FakeEndorser::default());
    let signer = FakeSigner::create().unwrap();
    let reference_values = test_reference_values();
    let clock = Arc::new(FakeClock { milliseconds_since_epoch: 0 });
    let mut actor = StorageActor::new(
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );
    assert_that!(actor.on_init(create_actor_context(true)), ok(anything()));

    create_client_session(
        &mut actor,
        b"session1",
        attester.clone(),
        endorser.clone(),
        signer.clone(),
        reference_values.clone(),
        clock.clone(),
    );

    // Creating a second session with a different ID should succeed.
    create_client_session(
        &mut actor,
        b"session2",
        attester,
        endorser,
        signer,
        reference_values,
        clock,
    );
}
