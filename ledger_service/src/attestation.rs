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

use alloc::string::String;
use anyhow::Context;
use cfc_crypto::CONFIG_PROPERTIES_CLAIM;
use core::time::Duration;
use coset::{cwt::ClaimName, cwt::ClaimsSet, CborSerializable, CoseKey, CoseSign1};
use federated_compute::proto::{
    value_matcher::Kind as ValueMatcherKind, value_matcher::NumberMatcher,
    value_matcher::StringMatcher, ApplicationMatcher, StructMatcher, ValueMatcher,
};
use oak_attestation_verification::verifier::{verify, verify_dice_chain};
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence, ReferenceValues};
use p256::ecdsa::{signature::Verifier, Signature, VerifyingKey};
use prost::Message;
use prost_types::{value::Kind as ValueKind, Struct, Value};

/// Various properties of an application running in an enclave.
#[derive(Debug, Default)]
pub struct Application<'a> {
    pub tag: &'a str,
    pub evidence: Option<&'a Evidence>,
    pub endorsements: Option<&'a Endorsements>,
    pub config_properties: Option<Struct>,
}

impl Application<'_> {
    /// Returns whether the application matches all conditions in the
    /// ApplicationMatcher.
    ///
    /// # Arguments
    ///
    /// * `matcher` - The matcher to match against. An empty or unset matcher
    ///   always matches.
    /// * `now` - The current time, represented as the duration since the Unix
    ///   epoch. (`std::time::Instant` is not no_std compatible.)
    ///
    /// # Return Value
    ///
    /// Returns a bool indicating whether the Application matches.
    pub fn matches(&self, matcher: &Option<ApplicationMatcher>, now: Duration) -> bool {
        let matcher = match matcher {
            Some(m) => m,
            None => return true, // An empty matcher matches everything.
        };
        self.tag_matches(&matcher.tag)
            && self.reference_values_match(&matcher.reference_values, now)
            && self.config_properties_match(&matcher.config_properties)
    }

    /// Returns whether the Application's tag matches the expected value.
    fn tag_matches(&self, tag: &Option<String>) -> bool {
        tag.as_ref().map_or(true, |t| self.tag == t)
    }

    /// Returns whether the Application's evidence and endorsements match the
    /// ReferenceValues.
    fn reference_values_match(
        &self,
        reference_values: &Option<ReferenceValues>,
        now: Duration,
    ) -> bool {
        reference_values.as_ref().map_or(true, |rv| {
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

    /// Returns whether the Application's config properties match the expected
    /// value.
    fn config_properties_match(&self, config_properties: &Option<StructMatcher>) -> bool {
        config_properties
            .as_ref()
            .map_or(true, |cp| Self::struct_value_matches(self.config_properties.as_ref(), cp))
    }

    /// Returns whether a Struct matches a StructMatcher.
    fn struct_value_matches(struct_value: Option<&Struct>, matcher: &StructMatcher) -> bool {
        matcher.fields.iter().all(|field_matcher| {
            // Traverse the struct based on the names in the FieldMatcher's path.
            let mut current_value: Option<&Value> = None;
            for path_part in field_matcher.path.split('.') {
                // Find the struct in which to look up the field. Initially (value == None),
                // this will be struct_value. Subsequently, it will be the value
                // found during the previous iteration.
                let s = match (&current_value, &struct_value) {
                    (None, Some(s)) => s,
                    (Some(Value { kind: Some(ValueKind::StructValue(ref s)) }), _) => s,
                    _ => return false,
                };
                current_value = s.fields.get(path_part);
            }

            current_value.map_or(false, |v| Self::value_matches(v, field_matcher.matcher.as_ref()))
        })
    }

    /// Returns whether a Value matches a ValueMatcher. All values match a
    /// missing or default ValueMatcher.
    fn value_matches(value: &Value, matcher: Option<&ValueMatcher>) -> bool {
        match (value, matcher) {
            (_, None | Some(ValueMatcher { kind: None })) => true,
            (
                Value { kind: Some(ValueKind::NumberValue(v)) },
                Some(ValueMatcher { kind: Some(ValueMatcherKind::NumberValue(m)) }),
            ) => Self::number_value_matches(*v, m),
            (
                Value { kind: Some(ValueKind::StringValue(v)) },
                Some(ValueMatcher { kind: Some(ValueMatcherKind::StringValue(m)) }),
            ) => Self::string_value_matches(v, m),
            _ => false,
        }
    }

    /// Returns whether a number matches a NumberMatcher.
    fn number_value_matches(value: f64, matcher: &NumberMatcher) -> bool {
        use federated_compute::proto::value_matcher::number_matcher::Kind;
        match matcher.kind {
            Some(Kind::Lt(x)) => value < x,
            Some(Kind::Le(x)) => value <= x,
            Some(Kind::Eq(x)) => value == x,
            Some(Kind::Ge(x)) => value >= x,
            Some(Kind::Gt(x)) => value > x,
            _ => false,
        }
    }

    /// Returns whether a string matches a StringMatcher.
    fn string_value_matches(value: &String, matcher: &StringMatcher) -> bool {
        use federated_compute::proto::value_matcher::string_matcher::Kind;
        match &matcher.kind {
            Some(Kind::Eq(x)) => x == value,
            _ => false,
        }
    }
}

/// Verifies enclave attestation and returns an Application describing its
/// properties.
///
/// Note that even if the verification succeeds, the attestation evidence should
/// not be trusted until it has been matched against reference values.
pub fn verify_attestation<'a>(
    public_key: &[u8],
    evidence: Option<&'a Evidence>,
    endorsements: Option<&'a Endorsements>,
    tag: &'a str,
) -> anyhow::Result<(Application<'a>, CoseKey)> {
    let mut config_properties = None;
    if let Some(evidence) = evidence {
        // If evidence was provided, pre-validate the DICE chain to ensure it's
        // structurally correct and that the public key is signed by its
        // application signing key. This duplicates validation that occurs
        // during `Application::matches`, but ensures that malformed/incomplete
        // requests are rejected earlier and with clearer error messages.
        let cwt = CoseSign1::from_slice(public_key)
            .map_err(anyhow::Error::msg)
            .context("invalid public key")?;
        if cwt.protected.header.alg.is_some()
            && cwt.protected.header.alg
                != Some(coset::Algorithm::Assigned(coset::iana::Algorithm::ES256))
        {
            return Err(anyhow::anyhow!(
                "unsupported public key algorithm: {:?}",
                cwt.protected.header.alg.unwrap()
            ));
        }
        let extracted_evidence = verify_dice_chain(evidence).context("invalid DICE chain")?;
        let verifying_key =
            VerifyingKey::from_sec1_bytes(&extracted_evidence.signing_public_key)
                .map_err(|err| anyhow::anyhow!("invalid application signing key: {:?}", err))?;
        cwt.verify_signature(b"", |signature, message| {
            verifying_key.verify(message, &Signature::from_slice(signature)?)
        })
        .map_err(anyhow::Error::msg)
        .context("invalid public key signature")?;

        // Extract the config properties. A missing claim results in config_properties =
        // None, which is not an error.
        config_properties = ClaimsSet::from_slice(cwt.payload.as_deref().unwrap_or_default())
            .map_err(anyhow::Error::msg)
            .and_then(|claims| {
                claims
                    .rest
                    .into_iter()
                    .find(|(name, _)| name == &ClaimName::PrivateUse(CONFIG_PROPERTIES_CLAIM))
                    .map(|(_, value)| {
                        value
                            .into_bytes()
                            .map_err(|err| anyhow::anyhow!("{:?}", err))
                            .and_then(|b| Struct::decode(b.as_slice()).map_err(anyhow::Error::msg))
                    })
                    .transpose()
            })
            .context("failed to decode config properties claim")?;
    }

    Ok((
        Application { tag, evidence, endorsements, config_properties },
        cfc_crypto::extract_key_from_cwt(public_key).context("invalid public key")?,
    ))
}

/// Helper function that returns a test Evidence message.
#[cfg(feature = "testing")]
pub fn get_test_evidence() -> Evidence {
    use oak_restricted_kernel_sdk::{attestation::EvidenceProvider, testing::MockEvidenceProvider};

    oak_attestation::dice::evidence_to_proto(
        MockEvidenceProvider::create().unwrap().get_evidence().clone(),
    )
    .unwrap()
}

/// Helper function that returns a test Endorsements message.
#[cfg(feature = "testing")]
pub fn get_test_endorsements() -> Endorsements {
    use oak_proto_rust::oak::attestation::v1::{
        endorsements, OakRestrictedKernelEndorsements, RootLayerEndorsements,
    };

    Endorsements {
        r#type: Some(endorsements::Type::OakRestrictedKernel(OakRestrictedKernelEndorsements {
            root_layer: Some(RootLayerEndorsements::default()),
            ..Default::default()
        })),
    }
}

/// Helper function that returns ReferenceValues that match the test Evidence.
#[cfg(feature = "testing")]
pub fn get_test_reference_values() -> oak_proto_rust::oak::attestation::v1::ReferenceValues {
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
                    ..Default::default()
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
    use alloc::{collections::BTreeMap, vec, vec::Vec};
    use cfc_crypto::PUBLIC_KEY_CLAIM;
    use coset::{
        cbor::Value, cwt::ClaimsSetBuilder, CborSerializable, CoseSign1Builder, HeaderBuilder,
    };
    use federated_compute::proto::{
        struct_matcher::FieldMatcher, value_matcher::number_matcher::Kind as NumberMatcherKind,
        value_matcher::string_matcher::Kind as StringMatcherKind,
    };
    use googletest::prelude::*;
    use oak_crypto::signer::Signer;
    use oak_proto_rust::oak::attestation::v1::{
        endorsements, OakRestrictedKernelEndorsements, ReferenceValues,
    };
    use oak_restricted_kernel_sdk::testing::MockSigner;

    /// Helper function to create a valid public key.
    fn create_public_key(config_properties: Option<&prost_types::Struct>) -> (Vec<u8>, CoseKey) {
        create_public_key_with_algorithm(config_properties, Some(coset::iana::Algorithm::ES256))
    }

    /// Helper function to create a public key with a specific algorithm.
    fn create_public_key_with_algorithm(
        config_properties: Option<&prost_types::Struct>,
        algorithm: Option<coset::iana::Algorithm>,
    ) -> (Vec<u8>, CoseKey) {
        let mut header = HeaderBuilder::new();
        if let Some(alg) = algorithm {
            header = header.algorithm(alg);
        }
        let (_, cose_key) = cfc_crypto::gen_keypair(b"key-id");
        let mut claims = ClaimsSetBuilder::new()
            .private_claim(PUBLIC_KEY_CLAIM, Value::from(cose_key.clone().to_vec().unwrap()));
        if let Some(cp) = config_properties {
            claims = claims.private_claim(CONFIG_PROPERTIES_CLAIM, Value::from(cp.encode_to_vec()));
        }
        let cwt = CoseSign1Builder::new()
            .protected(header.build())
            .payload(claims.build().to_vec().unwrap())
            .create_signature(b"", |message| MockSigner::create().unwrap().sign(message))
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
        let app = Application { tag: "tag", ..Default::default() };
        assert!(app.matches(
            &Some(ApplicationMatcher { tag: None, ..Default::default() }),
            Duration::default()
        ));
        assert!(app.matches(
            &Some(ApplicationMatcher { tag: Some(String::from("tag")), ..Default::default() }),
            Duration::default()
        ));
        assert!(!app.matches(
            &Some(ApplicationMatcher { tag: Some(String::from("other")), ..Default::default() }),
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
            &Some(ApplicationMatcher { reference_values: None, ..Default::default() }),
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

        // A matcher with reference values should not match an Application without
        // evidence or endorsements.
        assert!(!Application::default().matches(
            &Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            Duration::default()
        ));
    }

    #[test]
    fn test_application_matches_config_properties() {
        let app = Application {
            config_properties: Some(Struct {
                fields: BTreeMap::from([
                    ("x".into(), prost_types::Value { kind: Some(ValueKind::NumberValue(1.0)) }),
                    (
                        "y".into(),
                        prost_types::Value {
                            kind: Some(ValueKind::StringValue(String::from("apple"))),
                        },
                    ),
                ]),
            }),
            ..Default::default()
        };

        // An ApplicationMatcher that doesn't specify config_properties or specifies
        // empty config_properties should match.
        assert!(app.matches(&Some(ApplicationMatcher::default()), Duration::default(),));
        assert!(app.matches(
            &Some(ApplicationMatcher {
                config_properties: Some(StructMatcher::default()),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // If the FieldMatcher matches a NumberValue, the ApplicationMatcher should as
        // well.
        assert!(app.matches(
            &Some(ApplicationMatcher {
                config_properties: Some(StructMatcher {
                    fields: vec![FieldMatcher {
                        path: "x".into(),
                        matcher: Some(ValueMatcher {
                            kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                                kind: Some(NumberMatcherKind::Eq(1.0))
                            })),
                        }),
                    },],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // If the FieldMatcher matches a StringValue, the ApplicationMatcher should as
        // well.
        assert!(app.matches(
            &Some(ApplicationMatcher {
                config_properties: Some(StructMatcher {
                    fields: vec![FieldMatcher {
                        path: "y".into(),
                        matcher: Some(ValueMatcher {
                            kind: Some(ValueMatcherKind::StringValue(StringMatcher {
                                kind: Some(StringMatcherKind::Eq(String::from("apple")))
                            })),
                        }),
                    },],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // And matching should fail if the FieldMatcher doesn't match a NumberMatcher.
        assert!(!app.matches(
            &Some(ApplicationMatcher {
                config_properties: Some(StructMatcher {
                    fields: vec![FieldMatcher {
                        path: "x".into(),
                        matcher: Some(ValueMatcher {
                            kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                                kind: Some(NumberMatcherKind::Lt(1.0))
                            })),
                        }),
                    },],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            Duration::default(),
        ));

        // And matching should fail if the FieldMatcher doesn't match a StringMatcher.
        assert!(!app.matches(
            &Some(ApplicationMatcher {
                config_properties: Some(StructMatcher {
                    fields: vec![FieldMatcher {
                        path: "x".into(),
                        matcher: Some(ValueMatcher {
                            kind: Some(ValueMatcherKind::StringValue(StringMatcher {
                                kind: Some(StringMatcherKind::Eq(String::from("pear")))
                            })),
                        }),
                    },],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            Duration::default(),
        ));
    }

    #[test]
    fn test_verify_attestation() -> anyhow::Result<()> {
        let config_properties = Struct {
            fields: BTreeMap::from([(
                "x".into(),
                prost_types::Value { kind: Some(ValueKind::NumberValue(1.0)) },
            )]),
        };
        let (cwt, cose_key) = create_public_key(Some(&config_properties));
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
        assert_eq!(app.config_properties, Some(config_properties));
        assert_eq!(key, cose_key);
        anyhow::Ok(())
    }

    #[test]
    fn test_verify_attestation_without_evidence() -> anyhow::Result<()> {
        let (cwt, cose_key) = create_public_key(None);
        let tag = "tag";
        let (app, key) = verify_attestation(&cwt, None, None, tag)?;
        assert_eq!(app.tag, tag);
        assert_eq!(app.evidence, None);
        assert_eq!(app.endorsements, None);
        assert_eq!(key, cose_key);
        anyhow::Ok(())
    }

    #[test]
    fn test_verify_attestation_without_public_key_alg() {
        let (cwt, _) = create_public_key_with_algorithm(None, None);
        verify_attestation(&cwt, None, None, "tag").unwrap();
    }

    #[test]
    fn test_verify_attestation_without_config_properties() -> anyhow::Result<()> {
        let (cwt, _) = create_public_key(None);
        let (app, _) = verify_attestation(&cwt, None, None, "tag")?;
        assert_eq!(app.config_properties, None);
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
        let (cwt, _) = create_public_key(None);
        let evidence = Evidence::default();
        assert_that!(
            verify_attestation(&cwt, Some(&evidence), None, ""),
            err(displays_as(contains_substring("invalid DICE chain")))
        );
    }

    #[test]
    fn test_verify_attestation_invalid_public_key_alg() {
        let (cwt, _) = create_public_key_with_algorithm(None, Some(coset::iana::Algorithm::ES256K));
        assert_that!(
            verify_attestation(&cwt, Some(&get_test_evidence()), None, "tag"),
            err(displays_as(contains_substring("unsupported public key algorithm")))
        );
    }

    #[test]
    fn test_verify_attestation_invalid_public_key_signature() {
        let (cwt, _) = create_public_key(None);
        let mut invalid_cwt = CoseSign1::from_slice(&cwt).unwrap();
        invalid_cwt.signature = b"invalid".into();
        let invalid_public_key = invalid_cwt.to_vec().unwrap();
        assert_that!(
            verify_attestation(&invalid_public_key, Some(&get_test_evidence()), None, ""),
            err(displays_as(contains_substring("invalid public key signature")))
        );
    }

    #[test]
    fn test_struct_value_matches() {
        let value = Struct {
            fields: BTreeMap::from([
                ("1".into(), prost_types::Value { kind: Some(ValueKind::NumberValue(1.0)) }),
                (
                    "a".into(),
                    prost_types::Value {
                        kind: Some(ValueKind::StructValue(Struct {
                            fields: BTreeMap::from([
                                (
                                    "2".into(),
                                    prost_types::Value { kind: Some(ValueKind::NumberValue(2.0)) },
                                ),
                                (
                                    "b".into(),
                                    prost_types::Value {
                                        kind: Some(ValueKind::StructValue(Struct {
                                            fields: BTreeMap::from([(
                                                "3".into(),
                                                prost_types::Value {
                                                    kind: Some(ValueKind::NumberValue(3.0)),
                                                },
                                            )]),
                                        })),
                                    },
                                ),
                            ]),
                        })),
                    },
                ),
            ]),
        };

        // An empty StructMatcher always matches, even if the Struct is None.
        assert!(Application::struct_value_matches(Some(&value), &StructMatcher::default()));
        assert!(Application::struct_value_matches(None, &StructMatcher::default()));

        // But a missing Struct doesn't match a non-empty StructMatcher.
        assert!(!Application::struct_value_matches(
            None,
            &StructMatcher { fields: vec![FieldMatcher { path: "a".into(), matcher: None }] }
        ));

        // A missing or empty ValueMatcher checks presence.
        assert!(Application::struct_value_matches(
            Some(&value),
            &StructMatcher { fields: vec![FieldMatcher { path: "a".into(), matcher: None }] }
        ));
        assert!(Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "a".into(),
                    matcher: Some(ValueMatcher::default()),
                }],
            }
        ));
        assert!(!Application::struct_value_matches(
            Some(&value),
            &StructMatcher { fields: vec![FieldMatcher { path: "b".into(), matcher: None }] }
        ));
        assert!(!Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "b".into(),
                    matcher: Some(ValueMatcher::default()),
                }],
            }
        ));

        // Matches work on paths of different lengths.
        assert!(Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "1".into(),
                    matcher: Some(ValueMatcher {
                        kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                            kind: Some(NumberMatcherKind::Eq(1.0)),
                        })),
                    }),
                }],
            }
        ));
        assert!(Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "a.2".into(),
                    matcher: Some(ValueMatcher {
                        kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                            kind: Some(NumberMatcherKind::Eq(2.0)),
                        })),
                    }),
                }],
            }
        ));
        assert!(Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "a.b.3".into(),
                    matcher: Some(ValueMatcher {
                        kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                            kind: Some(NumberMatcherKind::Eq(3.0)),
                        })),
                    }),
                }],
            }
        ));

        // Matching fails if a path component doesn't exist.
        assert!(!Application::struct_value_matches(
            Some(&value),
            &StructMatcher {
                fields: vec![FieldMatcher {
                    path: "a.B.3".into(),
                    matcher: Some(ValueMatcher {
                        kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                            kind: Some(NumberMatcherKind::Eq(3.0)),
                        })),
                    }),
                }],
            }
        ));
    }

    #[test]
    fn test_value_matches() {
        // An unset or empty ValueMatcher always matches.
        assert!(Application::value_matches(&prost_types::Value::default(), None));
        assert!(Application::value_matches(
            &prost_types::Value::default(),
            Some(&ValueMatcher::default())
        ));
        assert!(Application::value_matches(
            &prost_types::Value { kind: Some(ValueKind::BoolValue(false)) },
            Some(&ValueMatcher::default())
        ));

        // Values and matchers of the same type can match.
        assert!(Application::value_matches(
            &prost_types::Value { kind: Some(ValueKind::NumberValue(0.0)) },
            Some(&ValueMatcher {
                kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                    kind: Some(NumberMatcherKind::Eq(0.0)),
                })),
            })
        ));
        assert!(!Application::value_matches(
            &prost_types::Value { kind: Some(ValueKind::NumberValue(0.0)) },
            Some(&ValueMatcher {
                kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                    kind: Some(NumberMatcherKind::Eq(1.0)),
                })),
            })
        ));

        // Values and matches of different types don't match.
        assert!(!Application::value_matches(
            &prost_types::Value { kind: Some(ValueKind::BoolValue(false)) },
            Some(&ValueMatcher {
                kind: Some(ValueMatcherKind::NumberValue(NumberMatcher {
                    kind: Some(NumberMatcherKind::Eq(0.0)),
                })),
            })
        ));
    }

    #[test]
    fn test_number_value_matches() {
        // Test each NumberMatcherKind's behavior for 0.9 ? 1.0, 1.0 ? 1.0, and 1.1 ?
        // 1.0.
        for (ctor, lt, eq, gt) in [
            ((|_x| None) as fn(f64) -> Option<NumberMatcherKind>, false, false, false),
            (|x| Some(NumberMatcherKind::Lt(x)), true, false, false),
            (|x| Some(NumberMatcherKind::Le(x)), true, true, false),
            (|x| Some(NumberMatcherKind::Eq(x)), false, true, false),
            (|x| Some(NumberMatcherKind::Ge(x)), false, true, true),
            (|x| Some(NumberMatcherKind::Gt(x)), false, false, true),
        ] {
            assert_eq!(
                Application::number_value_matches(0.9, &NumberMatcher { kind: ctor(1.0) }),
                lt
            );
            assert_eq!(
                Application::number_value_matches(1.0, &NumberMatcher { kind: ctor(1.0) }),
                eq
            );
            assert_eq!(
                Application::number_value_matches(1.1, &NumberMatcher { kind: ctor(1.0) }),
                gt
            );
        }
    }
}
