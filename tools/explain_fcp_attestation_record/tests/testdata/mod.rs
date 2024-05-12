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

use federated_compute::proto::data_access_policy::Transform;
use federated_compute::proto::struct_matcher::FieldMatcher;
use federated_compute::proto::value_matcher::NumberMatcher;
use federated_compute::proto::{
    access_budget, value_matcher, AttestationVerificationRecord, StructMatcher,
};
use federated_compute::proto::{AccessBudget, DataAccessPolicy};
use federated_compute::proto::{ApplicationMatcher, ValueMatcher};
use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    AmdSevReferenceValues, BinaryReferenceValue, Evidence, KernelBinaryReferenceValue,
    KernelLayerReferenceValues, OakRestrictedKernelReferenceValues, ReferenceValues,
    RootLayerReferenceValues, SkipVerification, TcbVersion, TextReferenceValue,
};
use prost::Message as _;

/// Returns an [`AttestationVerificationRecord`] with valid ledger attestation
/// evidence but with an empty data access policy.
pub fn record_with_empty_access_policy() -> AttestationVerificationRecord {
    let evidence = Evidence::decode(&include_bytes!("ledger_evidence.pb")[..])
        .expect("must be a valid evidence proto");
    AttestationVerificationRecord {
        attestation_evidence: Some(evidence),
        attestation_endorsements: None,
        data_access_policy: Some(DataAccessPolicy { ..Default::default() }),
    }
}

/// Returns an [`AttestationVerificationRecord`] with the same ledger
/// attestation evidence as [`record_with_empty_access_policy`] but with a data
/// access policy that contains a few transforms and access budgets. Most of the
/// transforms use Oak [`ReferenceValues`] which skip most checks, while one of
/// the transforms doesn't specify any reference values at all.
pub fn record_with_nonempty_access_policy() -> AttestationVerificationRecord {
    let mut record = record_with_empty_access_policy();

    // Let's populate the data access policy with some transforms and access
    // budgets, so the tool output for policies can be tested.
    let data_access_policy = record.data_access_policy.as_mut().unwrap();

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
        dest: Some(1),
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
        dest: None,
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
        dest: None,
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

    record
}

/// Creates a [`ReferenceValues`] instance that expects an Oak Restricted Kernel
/// application, skips all binary checks, but requires the attestation evidence
/// to be rooted in AMD SEV-SNP.
fn create_skip_all_amd_sev_reference_values() -> ReferenceValues {
    // A BinaryReferenceValue which skips all verifications.
    let binary_ref_value_skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification {})),
    };

    ReferenceValues {
        r#type: Some(reference_values::Type::OakRestrictedKernel(
            OakRestrictedKernelReferenceValues {
                root_layer: Some(RootLayerReferenceValues {
                    amd_sev: Some(AmdSevReferenceValues {
                        min_tcb_version: Some(TcbVersion {
                            boot_loader: 1,
                            tee: 2,
                            snp: 3,
                            microcode: 4,
                        }),
                        allow_debug: false,
                        stage0: Some(binary_ref_value_skip.clone()),
                    }),
                    ..Default::default()
                }),
                kernel_layer: Some(KernelLayerReferenceValues {
                    kernel: Some(KernelBinaryReferenceValue {
                        r#type: Some(kernel_binary_reference_value::Type::Skip(
                            SkipVerification {},
                        )),
                    }),
                    kernel_cmd_line_text: Some(TextReferenceValue {
                        r#type: Some(text_reference_value::Type::Skip(SkipVerification {})),
                    }),
                    init_ram_fs: Some(binary_ref_value_skip.clone()),
                    memory_map: Some(binary_ref_value_skip.clone()),
                    acpi: Some(binary_ref_value_skip.clone()),
                    ..Default::default()
                }),
                application_layer: Some(
                    oak_proto_rust::oak::attestation::v1::ApplicationLayerReferenceValues {
                        binary: Some(binary_ref_value_skip.clone()),
                        configuration: Some(binary_ref_value_skip.clone()),
                    },
                ),
            },
        )),
    }
}
