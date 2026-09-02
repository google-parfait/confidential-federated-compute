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

use access_policy_proto::fcp::confidentialcompute::{
    AccessBudget, ApplicationMatcher, DataAccessPolicy, StructMatcher, ValueMatcher, access_budget,
    data_access_policy::Transform, struct_matcher::FieldMatcher, value_matcher,
    value_matcher::NumberMatcher,
};
use access_policy_proto::reference_value_proto::oak::attestation::v1::{
    AmdSevReferenceValues, BinaryReferenceValue, ContainerLayerReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakContainersReferenceValues,
    ReferenceValues, RootLayerReferenceValues, SkipVerification, SystemLayerReferenceValues,
    TcbVersionReferenceValue, TextReferenceValue, binary_reference_value,
    kernel_binary_reference_value, reference_values, tcb_version_reference_value,
    text_reference_value,
};
use access_policy_proto::reference_value_proto::tcb_version_proto::oak::attestation::v1::TcbVersion;
use messages_proto::oak::session::v1::EndorsedEvidence;
use prost::Message as _;
use rustls::pki_types::pem::PemObject as _;
use sha2::{Digest as _, Sha256};
use signed_endorsements_proto::fcp::confidentialcompute::signed_endorsements::PipelineConfiguration;
use verification_record_proto::{
    fcp::confidentialcompute::AttestationVerificationRecord,
    payload_transparency_proto::fcp::confidentialcompute::{SignedPayload, signed_payload},
};

/// Helper to construct an [`AttestationVerificationRecord`] with
/// `SignedPayload`s and return the raw CAS payload bytes.
fn wrap_in_record(
    evidence_bytes: &[u8],
    access_policy: DataAccessPolicy,
) -> (AttestationVerificationRecord, Vec<u8>, Vec<u8>) {
    let access_policy_bytes = access_policy.encode_to_vec();
    let evidence_proto =
        messages_proto::evidence_proto::oak::attestation::v1::Evidence::decode(evidence_bytes)
            .expect("must be valid evidence proto");
    let endorsed_evidence_bytes =
        EndorsedEvidence { evidence: Some(evidence_proto), endorsements: None }.encode_to_vec();

    let record = AttestationVerificationRecord {
        encryption_key: Some(SignedPayload {
            signatures: vec![signed_payload::Signature {
                headers: signed_payload::signature::Headers {
                    oak_application_signature: Some(signed_payload::Signature {
                        headers: signed_payload::signature::Headers {
                            endorsed_evidence_sha256: Sha256::digest(&endorsed_evidence_bytes)
                                .to_vec(),
                            ..Default::default()
                        }
                        .encode_to_vec(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }
                .encode_to_vec(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        pipeline_configuration: Some(SignedPayload {
            payload: PipelineConfiguration {
                access_policy_sha256: Sha256::digest(&access_policy_bytes).to_vec(),
            }
            .encode_to_vec(),
            ..Default::default()
        }),
    };
    (record, access_policy_bytes, endorsed_evidence_bytes)
}

/// Returns an [`AttestationVerificationRecord`] with valid KMS attestation
/// evidence but with an empty data access policy.
pub fn record_with_empty_access_policy() -> (AttestationVerificationRecord, Vec<u8>, Vec<u8>) {
    wrap_in_record(
        include_bytes!("kms_evidence.binarypb"),
        DataAccessPolicy { ..Default::default() },
    )
}

/// Returns an [`AttestationVerificationRecord`] with the same KMS
/// attestation evidence as [`record_with_empty_access_policy`] but with a data
/// access policy that contains a few transforms and access budgets. Most of the
/// transforms use Oak [`ReferenceValues`] which skip most checks, while one of
/// the transforms doesn't specify any reference values at all.
pub fn record_with_nonempty_access_policy() -> (AttestationVerificationRecord, Vec<u8>, Vec<u8>) {
    let mut data_access_policy = DataAccessPolicy { ..Default::default() };

    // Define a few shared access budgets.
    //
    // This one will determine how often transform #3 can process transform #1's
    // output (it doesn't have to be a shared access budget, per se, but for the
    // test's sake it is).
    data_access_policy
        .shared_access_budgets
        .push(AccessBudget { kind: Some(access_budget::Kind::Times(5)) });
    // This one will determine how often transform #1 and #2 can process their
    // source blobs (at most 2 accesses may occur across both transforms).
    data_access_policy
        .shared_access_budgets
        .push(AccessBudget { kind: Some(access_budget::Kind::Times(2)) });

    // Next, define a few transforms.
    // Transform #1
    data_access_policy.transforms.push(Transform {
        // This is a transform that processes the initial input data and produces some output data
        // that can only be processed by transform #3.
        src: 0,
        application: Some(ApplicationMatcher {
            tag: Some("app2".to_string()),
            reference_values: Some(create_skip_all_amd_sev_reference_values()),
            config_properties: Some(StructMatcher {
                fields: vec![FieldMatcher {
                    path: "field_a".to_string(),
                    matcher: Some(ValueMatcher {
                        kind: Some(value_matcher::Kind::NumberValue(NumberMatcher {
                            kind: Some(value_matcher::number_matcher::Kind::Eq(1234.0)),
                        })),
                    }),
                }],
            }),
        }),
        // This transform can access its source blob twice, but that's only possible if transform
        // #2, which uses the same shared budget, doesn't use any of that budget.
        access_budget: Some(AccessBudget { kind: Some(access_budget::Kind::Times(2)) }),
        // This transform shares an access budget with transform #2.
        shared_access_budget_indices: vec![1],
    });

    // Transform #2
    data_access_policy.transforms.push(Transform {
        // This is a 'terminal' transform that processes the initial input data.
        src: 0,
        application: Some(ApplicationMatcher {
            tag: Some("app2".to_string()),
            reference_values: Some(create_skip_all_amd_sev_reference_values()),
            config_properties: Some(StructMatcher {
                fields: vec![FieldMatcher {
                    path: "field_b".to_string(),
                    matcher: Some(ValueMatcher {
                        kind: Some(value_matcher::Kind::NumberValue(NumberMatcher {
                            kind: Some(value_matcher::number_matcher::Kind::Eq(5678.0)),
                        })),
                    }),
                }],
            }),
        }),
        // No transform-specific access budget in this case, only a shared access budget.
        access_budget: None,
        // This transform shares an access budget with transform #1.
        shared_access_budget_indices: vec![1],
    });

    // Transform #3
    data_access_policy.transforms.push(Transform {
        // This is a 'terminal' transform which uses the output for the first transform.
        src: 1,
        application: Some(ApplicationMatcher {
            tag: Some("app3".to_string()),
            // For this last transform we purposely don't specify any reference values nor config
            // properaties, effectively letting any binary perform this transform.
            reference_values: None,
            config_properties: None,
        }),
        // No transform-specific access budget in this case, only a shared access budget (even
        // though the shared budget isn't actually shared with any other transforms).
        access_budget: None,
        shared_access_budget_indices: vec![0],
    });

    wrap_in_record(include_bytes!("kms_evidence.binarypb"), data_access_policy)
}

/// Returns the bytes of a legacy [`AttestationVerificationRecord`] (serialized
/// at commit `f5e8cfc6d3e20f3dcfe63ff8f8f11647ca2fd0f9`) that contains
/// unsupported Oak Restricted Kernel (ledger) evidence and no signed_payload
/// fields.
pub fn verification_record_with_ledger_evidence_without_signed_payload_bytes() -> &'static [u8] {
    include_bytes!("verification_record_with_ledger_evidence_without_signed_payload.binarypb")
}

/// Returns the bytes of a legacy [`AttestationVerificationRecord`] (serialized
/// at commit `f5e8cfc6d3e20f3dcfe63ff8f8f11647ca2fd0f9`) that contains Oak
/// Containers (KMS) evidence directly in legacy inline fields without
/// signed_payload fields.
pub fn verification_record_with_kms_evidence_without_signed_payload_bytes() -> &'static [u8] {
    include_bytes!("verification_record_with_kms_evidence_without_signed_payload.binarypb")
}

/// Returns a self-signed test root CA certificate and a rustls ServerConfig for
/// googleapis.com.
pub fn test_certs() -> (reqwest::Certificate, rustls::server::ServerConfig) {
    let ca_cert = reqwest::Certificate::from_pem(include_bytes!("test_root.pem"))
        .expect("must be a valid certificate");
    let server_config = rustls::server::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            vec![
                rustls::pki_types::CertificateDer::from_pem_slice(include_bytes!("test_cert.pem"))
                    .expect("must be a valid certificate"),
            ],
            rustls::pki_types::PrivateKeyDer::from_pem_slice(include_bytes!("test_cert.key.pem"))
                .expect("must be a valid private key"),
        )
        .expect("must be matching cert and private key");
    (ca_cert, server_config)
}

/// Creates a [`ReferenceValues`] instance that expects an Oak Containers
/// application, skips all binary checks, but requires the attestation evidence
/// to be rooted in AMD SEV-SNP.
fn create_skip_all_amd_sev_reference_values() -> ReferenceValues {
    // A BinaryReferenceValue which skips all verifications.
    let binary_ref_value_skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification {})),
    };

    ReferenceValues {
        r#type: Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues {
                amd_sev: Some(AmdSevReferenceValues {
                    milan: Some(TcbVersionReferenceValue {
                        r#type: Some(tcb_version_reference_value::Type::Minimum(TcbVersion {
                            boot_loader: 1,
                            tee: 2,
                            snp: 3,
                            microcode: 4,
                            fmc: 0,
                        })),
                    }),
                    allow_debug: false,
                    stage0: Some(binary_ref_value_skip.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            kernel_layer: Some(KernelLayerReferenceValues {
                kernel: Some(KernelBinaryReferenceValue {
                    r#type: Some(kernel_binary_reference_value::Type::Skip(SkipVerification {})),
                }),
                kernel_cmd_line_text: Some(TextReferenceValue {
                    r#type: Some(text_reference_value::Type::Skip(SkipVerification {})),
                }),
                init_ram_fs: Some(binary_ref_value_skip.clone()),
                memory_map: Some(binary_ref_value_skip.clone()),
                acpi: Some(binary_ref_value_skip.clone()),
                ..Default::default()
            }),
            system_layer: Some(SystemLayerReferenceValues {
                system_image: Some(binary_ref_value_skip.clone()),
                ..Default::default()
            }),
            container_layer: Some(ContainerLayerReferenceValues {
                binary: Some(binary_ref_value_skip.clone()),
                configuration: Some(binary_ref_value_skip.clone()),
            }),
        })),
    }
}
