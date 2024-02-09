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
use coset::CoseKey;
use federated_compute::proto::ApplicationMatcher;
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence};

/// Various properties of an application running in an enclave.
#[derive(Debug)]
pub struct Application<'a> {
    pub tag: &'a str,
}

impl Application<'_> {
    /// Returns whether the application matches all conditions in the ApplicationMatcher.
    pub fn matches(&self, matcher: &Option<ApplicationMatcher>) -> bool {
        let matcher = match matcher {
            Some(m) => m,
            None => return true, // An empty matcher matches everything.
        };
        matcher.tag.as_ref().map_or(true, |t| self.tag == t)
    }
}

/// Verifies enclave attestation and returns an Application describing its properties.
pub fn verify_attestation<'a>(
    public_key: &[u8],
    _attestation_evidence: &Option<Evidence>,
    _attestation_endorsements: &Option<Endorsements>,
    tag: &'a str,
) -> anyhow::Result<(Application<'a>, CoseKey)> {
    // TODO(b/288331695): Verify attestation.
    Ok((
        Application { tag },
        cfc_crypto::extract_key_from_cwt(public_key).context("invalid public key")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use cfc_crypto::PUBLIC_KEY_CLAIM;
    use coset::{cbor::Value, cwt::ClaimsSetBuilder, CborSerializable, CoseSign1Builder};
    use googletest::prelude::*;

    #[test]
    fn test_application_matches_empty_matcher() {
        assert!(Application { tag: "tag" }.matches(&None));
    }

    #[test]
    fn test_application_matches_tag() {
        let app = Application { tag: "tag" };
        assert!(app.matches(&Some(ApplicationMatcher {
            tag: None,
            ..Default::default()
        })));
        assert!(app.matches(&Some(ApplicationMatcher {
            tag: Some(String::from("tag")),
            ..Default::default()
        })));
        assert!(!app.matches(&Some(ApplicationMatcher {
            tag: Some(String::from("other")),
            ..Default::default()
        })));
    }

    #[test]
    fn test_verify_attestation() -> anyhow::Result<()> {
        let (_, cose_key) = cfc_crypto::gen_keypair(b"key-id");
        let public_key = CoseSign1Builder::new()
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
            .build()
            .to_vec()
            .unwrap();
        let tag = "tag";
        let (app, key) = verify_attestation(&public_key, &None, &None, tag)?;
        assert_eq!(app.tag, tag);
        assert_eq!(key, cose_key);
        anyhow::Ok(())
    }

    #[test]
    fn test_verify_attestation_invalid_key() {
        assert_that!(
            verify_attestation(b"invalid", &None, &None, "tag"),
            err(displays_as(contains_substring("invalid public key")))
        );
    }
}
