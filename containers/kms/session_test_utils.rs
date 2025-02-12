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
//
// Utilities for testing code that depends on oak_session.

use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification_types::util::Clock;
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence, ReferenceValues};
use oak_restricted_kernel_sdk::testing::MockAttester;
pub use oak_restricted_kernel_sdk::testing::MockSigner as FakeSigner;

/// Creates test reference values compatible with `FakeAttester` and
/// `FakeEndorser`.
pub fn test_reference_values() -> ReferenceValues {
    use oak_proto_rust::oak::attestation::v1::{
        binary_reference_value, kernel_binary_reference_value, reference_values,
        text_reference_value, ApplicationLayerReferenceValues, BinaryReferenceValue,
        InsecureReferenceValues, KernelBinaryReferenceValue, KernelLayerReferenceValues,
        OakRestrictedKernelReferenceValues, RootLayerReferenceValues, SkipVerification,
        TextReferenceValue,
    };

    let skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
    };
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
                        r#type: Some(text_reference_value::Type::Skip(SkipVerification::default())),
                    }),
                    init_ram_fs: Some(skip.clone()),
                    memory_map: Some(skip.clone()),
                    acpi: Some(skip.clone()),
                }),
                application_layer: Some(ApplicationLayerReferenceValues {
                    binary: Some(skip.clone()),
                    configuration: Some(skip.clone()),
                }),
            },
        )),
    }
}

/// A fake Attester that returns insecure Evidence.
#[derive(Clone)]
pub struct FakeAttester {
    evidence: Evidence,
}
impl FakeAttester {
    pub fn create() -> anyhow::Result<Self> {
        Ok(FakeAttester { evidence: MockAttester::create()?.quote()? })
    }
}
impl Attester for FakeAttester {
    fn extend(&mut self, _encoded_event: &[u8]) -> anyhow::Result<()> {
        anyhow::bail!("FakeAttester::extend is not implemented");
    }
    fn quote(&self) -> anyhow::Result<Evidence> {
        Ok(self.evidence.clone())
    }
}

/// A fake Endorser that produces empty Endorsements.
#[derive(Clone, Default)]
pub struct FakeEndorser {}
impl Endorser for FakeEndorser {
    fn endorse(&self, _evidence: Option<&Evidence>) -> anyhow::Result<Endorsements> {
        use oak_proto_rust::oak::attestation::v1::{
            endorsements, OakRestrictedKernelEndorsements, RootLayerEndorsements,
        };
        Ok(Endorsements {
            r#type: Some(endorsements::Type::OakRestrictedKernel(
                OakRestrictedKernelEndorsements {
                    // root_layer is required.
                    root_layer: Some(RootLayerEndorsements::default()),
                    ..Default::default()
                },
            )),
            ..Default::default()
        })
    }
}

/// A fake Clock that returns a fixed time.
pub struct FakeClock {
    pub milliseconds_since_epoch: i64,
}
impl Clock for FakeClock {
    fn get_milliseconds_since_epoch(&self) -> i64 {
        self.milliseconds_since_epoch
    }
}
