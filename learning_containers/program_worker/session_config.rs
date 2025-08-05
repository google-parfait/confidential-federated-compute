// Copyright 2025 The Trusted Computations Platform Authors.
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
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_proto_rust::oak::session::v1::EndorsedEvidence;
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::InstanceSessionBinder;
use oak_session::session_binding::SessionBinder;
use oak_session::{attestation::AttestationType, config::SessionConfig, handshake::HandshakeType};
use prost::Message;
use std::sync::Arc;
use tokio::runtime::Runtime;

const SESSION_ID: &str = "cfc_program_worker";

/// Creates a SessionConfig for the program worker.
///
/// This function is exported to C so that it can be called from the program
/// worker binary.
///
/// # Arguments
///
/// * `endorsed_evidence_bytes`: A serialized EndorsedEvidence proto.
/// * `endorsed_evidence_len`: The length of the serialized EndorsedEvidence
///   proto.
///
/// # Returns
///
/// A raw pointer to a SessionConfig. The caller is responsible for managing the
/// memory of this object.
///
/// # Safety
///
/// The returned config is an opaque raw pointer to a `SessionConfig` object.
/// The handle is intended to be reclaimed by passing it the FFI factory of an
/// OakSession object like `oak_session::ffi::new_client_session`.
///
/// `endorsed_evidence_bytes` and `endorsed_evidence_len` must describe a valid
/// buffer. Data must not be modified during this function call. It may be
/// modified or discarded after, as this function will make its own copy.
#[no_mangle]
pub unsafe extern "C" fn create_session_config(
    endorsed_evidence_bytes: *const u8,
    endorsed_evidence_len: usize,
) -> *mut SessionConfig {
    let runtime = Runtime::new().expect("Failed to create Tokio runtime");
    let channel = runtime
        .block_on(oak_sdk_containers::default_orchestrator_channel())
        .expect("Failed to create orchestrator channel");

    let endorsed_evidence_slice =
        std::slice::from_raw_parts(endorsed_evidence_bytes, endorsed_evidence_len);
    let endorsed_evidence =
        EndorsedEvidence::decode(endorsed_evidence_slice).expect("failed to decode evidence");
    let evidence = endorsed_evidence.evidence.expect("EndorsedEvidence.evidence not set");
    let endorsements =
        endorsed_evidence.endorsements.expect("EndorsedEvidence.endorsements not set");
    let attester: Arc<dyn Attester> = Arc::new(StaticAttester::new(evidence.clone()));
    let endorser: Arc<dyn Endorser> = Arc::new(StaticEndorser::new(endorsements.clone()));
    let session_binder: Arc<dyn SessionBinder> = Arc::new(InstanceSessionBinder::create(&channel));

    let builder =
        SessionConfig::builder(AttestationType::SelfUnidirectional, HandshakeType::NoiseNN)
            .add_self_attester_ref(SESSION_ID.into(), &attester)
            .add_self_endorser_ref(SESSION_ID.into(), &endorser)
            .add_session_binder_ref(SESSION_ID.into(), &session_binder);
    Box::into_raw(Box::new(builder.build()))
}
