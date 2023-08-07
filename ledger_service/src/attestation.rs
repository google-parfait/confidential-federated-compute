// Copyright 2023 The Confidential Federated Compute Authors.
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

use federated_compute::proto::ApplicationMatcher;

/// Various properties of an application running in an enclave.
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
    _public_key: &[u8],
    _attestation: &[u8],
    tag: &'a str,
) -> Result<Application<'a>, micro_rpc::Status> {
    // TODO(b/288331695): Verify attestation.
    Ok(Application { tag })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

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
    fn test_verify_attestation() -> Result<(), micro_rpc::Status> {
        let tag = "tag";
        let app = verify_attestation(b"", b"", tag)?;
        assert_eq!(app.tag, tag);
        micro_rpc::Ok(())
    }
}
