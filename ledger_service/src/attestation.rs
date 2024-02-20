// Copyright 2023 Google LLC.
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

extern crate alloc;

use anyhow::Context;
use core::time::Duration;
use coset::{CborSerializable, CoseKey, CoseSign1};
use federated_compute::proto::ApplicationMatcher;
use oak_attestation_verification::verifier::{verify, verify_dice_chain};
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence};
use p256::ecdsa::{signature::Verifier, Signature, VerifyingKey};

/// Various properties of an application running in an enclave.
#[derive(Debug, Default)]
pub struct Application<'a> {
    pub tag: &'a str,
    pub evidence: Option<&'a Evidence>,
    pub endorsements: Option<&'a Endorsements>,
}

impl Application<'_> {
    /// Returns whether the application matches all conditions in the ApplicationMatcher.
    ///
    /// # Arguments
    ///
    /// * `matcher` - The matcher to match against. An empty or unset matcher always matches.
    /// * `now` - The current time, represented as the duration since the Unix epoch.
    ///   (`std::time::Instant` is not no_std compatible.)
    ///
    /// # Return Value
    ///
    /// Returns a bool indicating whether the Application matches.
    pub fn matches(&self, matcher: &Option<ApplicationMatcher>, now: Duration) -> bool {
        let matcher = match matcher {
            Some(m) => m,
            None => return true, // An empty matcher matches everything.
        };
        matcher.tag.as_ref().map_or(true, |t| self.tag == t)
            && matcher.reference_values.as_ref().map_or(true, |rv| {
                let now_utc_millis = match now.as_millis().try_into() {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                let (evidence, endorsements) = match (self.evidence, self.endorsements) {
                    (Some(evidence), Some(endorsements)) => (evidence, endorsements),
                    _ => return false,
                };
                verify(now_utc_millis, evidence, endorsements, rv).is_ok()
            })
    }
}

/// Verifies enclave attestation and returns an Application describing its properties.
///
/// Note that even if the verification succeeds, the attestation evidence should not be trusted
/// until it has been matched against reference values.
pub fn verify_attestation<'a>(
    public_key: &[u8],
    evidence: Option<&'a Evidence>,
    endorsements: Option<&'a Endorsements>,
    tag: &'a str,
) -> anyhow::Result<(Application<'a>, CoseKey)> {
    if let Some(evidence) = evidence {
        // If evidence was provided, pre-validate the DICE chain to ensure it's structurally
        // correct and that the public key is signed by its application signing key. This
        // duplicates validation that occurs during `Application::matches`, but ensures that
        // malformed/incomplete requests are rejected earlier and with clearer error messages.
        let cwt = CoseSign1::from_slice(public_key)
            .map_err(anyhow::Error::msg)
            .context("invalid public key")?;
        let extracted_evidence = verify_dice_chain(evidence).context("invalid DICE chain")?;
        let verifying_key =
            VerifyingKey::from_sec1_bytes(&extracted_evidence.signing_public_key)
                .map_err(|err| anyhow::anyhow!("invalid application signing key: {:?}", err))?;
        cwt.verify_signature(b"", |signature, message| {
            verifying_key.verify(message, &Signature::from_slice(signature)?)
        })
        .map_err(|err| anyhow::anyhow!("invalid public key signature: {:?}", err))?;
    }

    Ok((
        Application {
            tag,
            evidence,
            endorsements,
        },
        cfc_crypto::extract_key_from_cwt(public_key).context("invalid public key")?,
    ))
}

/// Helper function that returns a test Evidence message.
#[cfg(test)]
pub fn get_test_evidence() -> Evidence {
    use oak_restricted_kernel_sdk::{mock_attestation::MockEvidenceProvider, EvidenceProvider};

    oak_attestation::dice::evidence_to_proto(
        MockEvidenceProvider::create()
            .unwrap()
            .get_evidence()
            .clone(),
    )
    .unwrap()
}

/// Helper function that returns a test Endorsements message.
#[cfg(test)]
pub fn get_test_endorsements() -> Endorsements {
    use oak_proto_rust::oak::attestation::v1::{
        endorsements, OakRestrictedKernelEndorsements, RootLayerEndorsements,
    };

    Endorsements {
        r#type: Some(endorsements::Type::OakRestrictedKernel(
            OakRestrictedKernelEndorsements {
                root_layer: Some(RootLayerEndorsements::default()),
                ..Default::default()
            },
        )),
    }
}

/// Helper function that returns ReferenceValues that match the test Evidence.
#[cfg(test)]
pub fn get_test_reference_values() -> oak_proto_rust::oak::attestation::v1::ReferenceValues {
    use oak_proto_rust::oak::attestation::v1::{
        binary_reference_value, reference_values, ApplicationLayerReferenceValues,
        BinaryReferenceValue, InsecureReferenceValues, KernelLayerReferenceValues,
        OakRestrictedKernelReferenceValues, ReferenceValues, RootLayerReferenceValues,
        SkipVerification,
    };

    let skip = BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(
            SkipVerification::default(),
        )),
    };
    ReferenceValues {
        r#type: Some(reference_values::Type::OakRestrictedKernel(
            OakRestrictedKernelReferenceValues {
                root_layer: Some(RootLayerReferenceValues {
                    insecure: Some(InsecureReferenceValues::default()),
                    ..Default::default()
                }),
                kernel_layer: Some(KernelLayerReferenceValues {
                    kernel_image: Some(skip.clone()),
                    kernel_cmd_line: Some(skip.clone()),
                    kernel_setup_data: Some(skip.clone()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::String, vec::Vec};
    use cfc_crypto::PUBLIC_KEY_CLAIM;
    use coset::{cbor::Value, cwt::ClaimsSetBuilder, CborSerializable, CoseSign1Builder};
    use googletest::prelude::*;
    use oak_proto_rust::oak::attestation::v1::{
        endorsements, OakRestrictedKernelEndorsements, ReferenceValues,
    };
    use oak_restricted_kernel_sdk::{mock_attestation::MockSigner, Signer};

    /// Helper function to create a valid public key.
    fn create_public_key() -> (Vec<u8>, CoseKey) {
        let (_, cose_key) = cfc_crypto::gen_keypair(b"key-id");
        let cwt = CoseSign1Builder::new()
            .payload(
                ClaimsSetBuilder::new()
                    .private_claim(
                        PUBLIC_KEY_CLAIM,
                        Value::from(cose_key.clone().to_vec().unwrap()),
                    )
                    .build()
                    .to_vec()
                    .unwrap(),
            )
            .create_signature(b"", |message| {
                MockSigner::create()
                    .unwrap()
                    .sign(message)
                    .unwrap()
                    .signature
            })
            .build()
            .to_vec()
            .unwrap();
        (cwt, cose_key)
    }

    #[test]
    fn test_application_matches_empty_matcher() {
        assert!(Application::default().matches(&None, Duration::default()));
    }

    #[test]
    fn test_application_matches_tag() {
        let app = Application {
            tag: "tag",
            ..Default::default()
        };
        assert!(app.matches(
            &Some(ApplicationMatcher {
                tag: None,
                ..Default::default()
            }),
            Duration::default()
        ));
        assert!(app.matches(
            &Some(ApplicationMatcher {
                tag: Some(String::from("tag")),
                ..Default::default()
            }),
            Duration::default()
        ));
        assert!(!app.matches(
            &Some(ApplicationMatcher {
                tag: Some(String::from("other")),
                ..Default::default()
            }),
            Duration::default()
        ));
    }

    #[test]
    fn test_application_matches_attestation() {
        let evidence = get_test_evidence();
        let endorsements = get_test_endorsements();
        let app = Application {
            evidence: Some(&evidence),
            endorsements: Some(&endorsements),
            ..Default::default()
        };
        assert!(app.matches(
            &Some(ApplicationMatcher {
                reference_values: None,
                ..Default::default()
            }),
            Duration::default()
        ));

        // Valid reference values should match.
        assert!(app.matches(
            &Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // Empty reference values will cause validation to fail.
        assert!(!app.matches(
            &Some(ApplicationMatcher {
                reference_values: Some(ReferenceValues::default()),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // A matcher with reference values should not match an Application without evidence or
        // endorsements.
        assert!(!Application::default().matches(
            &Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            Duration::default()
        ));
    }

    #[test]
    fn test_verify_attestation() -> anyhow::Result<()> {
        let (cwt, cose_key) = create_public_key();
        let evidence = get_test_evidence();
        let endorsements = Endorsements {
            r#type: Some(endorsements::Type::OakRestrictedKernel(
                OakRestrictedKernelEndorsements::default(),
            )),
        };
        let tag = "tag";
        let (app, key) = verify_attestation(&cwt, Some(&evidence), Some(&endorsements), tag)?;
        assert_eq!(app.tag, tag);
        assert_eq!(app.evidence, Some(&evidence));
        assert_eq!(app.endorsements, Some(&endorsements));
        assert_eq!(key, cose_key);
        anyhow::Ok(())
    }

    #[test]
    fn test_verify_attestation_without_evidence() -> anyhow::Result<()> {
        let (cwt, cose_key) = create_public_key();
        let tag = "tag";
        let (app, key) = verify_attestation(&cwt, None, None, tag)?;
        assert_eq!(app.tag, tag);
        assert_eq!(key, cose_key);
        anyhow::Ok(())
    }

    #[test]
    fn test_verify_attestation_invalid_key() {
        assert_that!(
            verify_attestation(b"invalid", None, None, "tag"),
            err(displays_as(contains_substring("invalid public key")))
        );
    }

    #[test]
    fn test_verify_attestation_invalid_evidence() {
        let (cwt, _) = create_public_key();
        let evidence = Evidence::default();
        assert_that!(
            verify_attestation(&cwt, Some(&evidence), None, ""),
            err(displays_as(contains_substring("invalid DICE chain")))
        );
    }

    #[test]
    fn test_verify_attestation_invalid_public_key_signature() {
        let (cwt, _) = create_public_key();
        let mut invalid_cwt = CoseSign1::from_slice(&cwt).unwrap();
        invalid_cwt.signature = b"invalid".into();
        let invalid_public_key = invalid_cwt.to_vec().unwrap();
        assert_that!(
            verify_attestation(&invalid_public_key, Some(&get_test_evidence()), None, ""),
            err(displays_as(contains_substring(
                "invalid public key signature"
            )))
        );
    }
}
