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

use std::sync::Arc;

use access_policy_proto::reference_value_proto::oak::attestation::v1::ReferenceValues;
use kms_proto::{
    endorsement_proto::oak::attestation::v1::Endorsements,
    evidence_proto::oak::attestation::v1::Evidence,
};
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification_types::util::Clock;
use oak_sdk_common::{StaticAttester, StaticEncryptionKeyHandle, StaticEndorser};
use oak_sdk_standalone::Standalone;
use p256::ecdsa::{SigningKey, VerifyingKey};
use prost_proto_conversion::ProstProtoConversionExt;
use rand_core::OsRng;

pub use p256::ecdsa::SigningKey as TestSigner;

static FAKE_DATA: std::sync::LazyLock<(Standalone, SigningKey)> = std::sync::LazyLock::new(|| {
    let signing_key = SigningKey::random(&mut OsRng);
    let standalone = Standalone::builder()
        .signing_key_pair(Some((signing_key.clone(), VerifyingKey::from(&signing_key))))
        .build()
        .expect("failed to build Standalone");
    (standalone, signing_key)
});

/// Returns test evidence that uses an insecure root.
pub fn get_test_evidence() -> Evidence {
    FAKE_DATA.0.endorsed_evidence().evidence.unwrap().convert().unwrap()
}

/// Returns an Attester that uses the test evidence.
pub fn get_test_attester() -> Arc<dyn Attester> {
    Arc::new(StaticAttester::new(FAKE_DATA.0.endorsed_evidence().evidence.unwrap()))
}

/// Returns test endorsements.
pub fn get_test_endorsements() -> Endorsements {
    FAKE_DATA.0.endorsed_evidence().endorsements.unwrap().convert().unwrap()
}

/// Returns an Endorser that uses the test endorsements.
pub fn get_test_endorser() -> Arc<dyn Endorser> {
    Arc::new(StaticEndorser::new(FAKE_DATA.0.endorsed_evidence().endorsements.unwrap()))
}

/// Returns an EncryptionKeyHandle that uses the test evidence's encryption key.
pub fn get_test_encryption_key_handle() -> StaticEncryptionKeyHandle {
    FAKE_DATA.0.encryption_key_handle()
}

/// Returns a Signer that uses the test evidence's signing key.
pub fn get_test_signer() -> TestSigner {
    FAKE_DATA.1.clone()
}

/// Returns reference values compatible with the test evidence.
pub fn get_test_reference_values() -> ReferenceValues {
    use access_policy_proto::reference_value_proto::oak::attestation::v1::{
        binary_reference_value, kernel_binary_reference_value, reference_values,
        text_reference_value, BinaryReferenceValue, ContainerLayerReferenceValues,
        InsecureReferenceValues, KernelBinaryReferenceValue, KernelLayerReferenceValues,
        OakContainersReferenceValues, RootLayerReferenceValues, SkipVerification,
        SystemLayerReferenceValues, TextReferenceValue,
    };

    let skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
    };
    ReferenceValues {
        r#type: Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
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
            system_layer: Some(SystemLayerReferenceValues { system_image: Some(skip.clone()) }),
            container_layer: Some(ContainerLayerReferenceValues {
                binary: Some(skip.clone()),
                configuration: Some(skip.clone()),
            }),
        })),
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
