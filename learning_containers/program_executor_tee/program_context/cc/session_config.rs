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

use std::sync::Arc;

use anyhow::{bail, Result};
use oak_attestation_verification::{
    policy::{
        container::ContainerPolicy, firmware::FirmwarePolicy, kernel::KernelPolicy,
        platform::AmdSevSnpPolicy, system::SystemPolicy,
    },
    verifier::{AmdSevSnpDiceAttestationVerifier, EventLogVerifier},
};
use oak_attestation_verification_types::verifier::AttestationVerifier;
use oak_proto_rust::oak::attestation::v1::{
    reference_values, AmdSevReferenceValues, OakContainersReferenceValues, ReferenceValues,
    RootLayerReferenceValues,
};
use oak_session::{
    attestation::AttestationType, config::SessionConfig, handshake::HandshakeType,
    key_extractor::DefaultBindingKeyExtractor,
};
use oak_time_std::clock::SystemTimeClock;
use prost::Message;

const SESSION_ID: &str = "cfc_program_worker";

fn create_session_config_internal(reference_values: ReferenceValues) -> Result<SessionConfig> {
    let peer_verifier: Box<dyn AttestationVerifier> = match &reference_values.r#type {
        // Oak Containers (insecure)
        Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues { insecure: Some(_), .. }),
            kernel_layer: Some(kernel_ref_vals),
            system_layer: Some(system_ref_vals),
            container_layer: Some(container_ref_vals),
        })) => {
            // TODO: b/432726860 - use InsecureDiceAttestationVerifier once it's available.
            Box::new(EventLogVerifier::new(
                vec![
                    Box::new(KernelPolicy::new(kernel_ref_vals)),
                    Box::new(SystemPolicy::new(system_ref_vals)),
                    Box::new(ContainerPolicy::new(container_ref_vals)),
                ],
                Arc::new(SystemTimeClock {}),
            ))
        }

        // Oak Containers (SEV-SNP)
        Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer:
                Some(RootLayerReferenceValues {
                    amd_sev:
                        Some(
                            amd_sev_ref_vals @ AmdSevReferenceValues {
                                stage0: Some(stage0_ref_vals),
                                ..
                            },
                        ),
                    insecure: None,
                    ..
                }),
            kernel_layer: Some(kernel_ref_vals),
            system_layer: Some(system_ref_vals),
            container_layer: Some(container_ref_vals),
        })) => Box::new(AmdSevSnpDiceAttestationVerifier::new(
            AmdSevSnpPolicy::new(amd_sev_ref_vals),
            Box::new(FirmwarePolicy::new(stage0_ref_vals)),
            vec![
                Box::new(KernelPolicy::new(kernel_ref_vals)),
                Box::new(SystemPolicy::new(system_ref_vals)),
                Box::new(ContainerPolicy::new(container_ref_vals)),
            ],
            Arc::new(SystemTimeClock {}),
        )),

        _ => {
            bail!("unsupported reference values");
        }
    };

    let builder =
        SessionConfig::builder(AttestationType::PeerUnidirectional, HandshakeType::NoiseNN)
            .add_peer_verifier_with_key_extractor(
                SESSION_ID.into(),
                peer_verifier,
                Box::new(DefaultBindingKeyExtractor {}),
            );

    Ok(builder.build())
}

/// Creates a SessionConfig for the program executor.
///
/// This function is exported to C so that it can be called from the program
/// executor binary.
///
/// # Arguments
///
/// * `reference_values_bytes`: A serialized ReferenceValues proto.
/// * `reference_values_len`: The length of the serialized ReferenceValues
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
/// `reference_values_bytes` and `reference_values_len` must describe a valid
/// buffer. Data must not be modified during this function call. It may be
/// modified or discarded after, as this function will make its own copy.
#[no_mangle]
pub unsafe extern "C" fn create_session_config(
    reference_values_bytes: *const u8,
    reference_values_len: usize,
) -> *mut SessionConfig {
    let reference_values_slice =
        std::slice::from_raw_parts(reference_values_bytes, reference_values_len);
    let reference_values = match ReferenceValues::decode(reference_values_slice) {
        Ok(rv) => rv,
        Err(_) => return std::ptr::null_mut(),
    };

    match create_session_config_internal(reference_values) {
        Ok(config) => Box::into_raw(Box::new(config)),
        Err(_) => std::ptr::null_mut(),
    }
}
