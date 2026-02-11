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

use anyhow::bail;
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification::{
    AmdSevSnpDiceAttestationVerifier, AmdSevSnpPolicy, ContainerPolicy, FirmwarePolicy,
    InsecureAttestationVerifier, KernelPolicy, SystemPolicy,
};
use oak_attestation_verification_types::verifier::AttestationVerifier;
use oak_crypto::{encryptor::Encryptor, noise_handshake::OrderedCrypter};
use oak_proto_rust::oak::attestation::v1::{
    reference_values, AmdSevReferenceValues, OakContainersReferenceValues, ReferenceValues,
    RootLayerReferenceValues,
};
use oak_session::{
    aggregators::PassThrough,
    attestation::AttestationType,
    config::{EncryptorProvider, SessionConfig},
    encryptors::UnorderedChannelEncryptor,
    handshake::HandshakeType,
    key_extractor::DefaultBindingKeyExtractor,
    session_binding::{SessionBinder, SignatureBindingVerifierProvider},
};
use oak_session_endorsed_evidence::{
    EndorsedEvidenceBindableAssertionGenerator, EndorsedEvidenceBoundAssertionVerifier,
};
use oak_time::Clock;

const ASSERTION_ID: &str = "cfc_kms_assertion";
const SESSION_ID: &str = "cfc_kms";

struct UnorderedEncryptorProvider;
impl EncryptorProvider for UnorderedEncryptorProvider {
    fn provide_encryptor(
        &self,
        crypter: OrderedCrypter,
    ) -> Result<Box<dyn Encryptor>, anyhow::Error> {
        TryInto::<UnorderedChannelEncryptor>::try_into((crypter, /* window_size= */ 64))
            .map(|v| Box::new(v) as Box<dyn Encryptor>)
    }
}

/// Creates a new SessionConfig for connections between KMS servers.
pub fn create_session_config(
    attester: &Arc<dyn Attester>,
    endorser: &Arc<dyn Endorser>,
    session_binder: &Arc<dyn SessionBinder>,
    reference_values: &ReferenceValues,
    clock: Arc<dyn Clock>,
) -> anyhow::Result<SessionConfig> {
    let peer_verifier: Box<dyn AttestationVerifier> = match &reference_values.r#type {
        // Oak Containers (insecure)
        Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues { insecure: Some(_), .. }),
            kernel_layer: Some(kernel_ref_vals),
            system_layer: Some(system_ref_vals),
            container_layer: Some(container_ref_vals),
        })) => Box::new(InsecureAttestationVerifier::new(
            clock,
            vec![
                Box::new(KernelPolicy::new(kernel_ref_vals)),
                Box::new(SystemPolicy::new(system_ref_vals)),
                Box::new(ContainerPolicy::new(container_ref_vals)),
            ],
        )),

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
            clock,
        )),

        _ => bail!("unsupported ReferenceValues"),
    };

    Ok(SessionConfig::builder(AttestationType::Bidirectional, HandshakeType::NoiseNN)
        .add_self_attester_ref(SESSION_ID.into(), attester)
        .add_self_endorser_ref(SESSION_ID.into(), endorser)
        .add_self_assertion_generator(
            String::from(ASSERTION_ID),
            Box::new(EndorsedEvidenceBindableAssertionGenerator::new(
                attester.clone(),
                endorser.clone(),
                session_binder.clone(),
            )),
        )
        .add_peer_assertion_verifier(
            String::from(ASSERTION_ID),
            Box::new(EndorsedEvidenceBoundAssertionVerifier::new(
                peer_verifier.into(),
                Arc::new(SignatureBindingVerifierProvider::new(Arc::new(
                    DefaultBindingKeyExtractor {},
                ))),
            )),
        )
        // Since only one assertion type is used, a trivial PassThrough
        // aggregator is sufficient.
        .set_assertion_attestation_aggregator(Box::new(PassThrough {}))
        // The communication channel is not guaranteed to be ordered.
        .set_encryption_provider(Box::new(UnorderedEncryptorProvider))
        .add_session_binder_ref(SESSION_ID.into(), session_binder)
        .build())
}
