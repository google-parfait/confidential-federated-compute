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

use crate::attestation::Application;
use federated_compute::proto::DataAccessPolicy;

/// A BudgetTracker keeps track of the remaining budgets for zero or more blobs.
#[derive(Default)]
pub struct BudgetTracker {}

impl BudgetTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds the first matching transform in the policy that has sufficient budget available.
    pub fn find_matching_transform(
        &self,
        _blob_id: &[u8],
        node_id: u32,
        policy: &DataAccessPolicy,
        app: &Application,
    ) -> Result<usize, micro_rpc::Status> {
        for (i, transform) in policy.transforms.iter().enumerate() {
            if transform.src != node_id || !app.matches(&transform.application) {
                continue;
            }

            // TODO(b/288282266): Check that there's budget available.
            return Ok(i);
        }

        Err(micro_rpc::Status::new_with_message(
            micro_rpc::StatusCode::FailedPrecondition,
            "requesting application does not match the access policy",
        ))
    }

    /// Updates the budget for a blob to reflect a new access.
    pub fn update_budget(
        &mut self,
        _blob_id: &[u8],
        _transform_index: usize,
        _policy: &DataAccessPolicy,
    ) -> Result<(), micro_rpc::Status> {
        // TODO(b/288282266): Update the budget.
        Ok(())
    }

    /// Consumes all remaining budget for a blob, making all future calls to update_budget fail.
    pub fn consume_budget(&mut self, _blob_id: &[u8]) -> Result<(), micro_rpc::Status> {
        // TODO(b/288282266): Clear the budget.
        Err(micro_rpc::Status::new(micro_rpc::StatusCode::Unimplemented))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::{borrow::ToOwned, vec};
    use federated_compute::proto::{data_access_policy::Transform, ApplicationMatcher};

    #[test]
    fn test_find_matching_transform_success() {
        let tracker = BudgetTracker::default();
        let app = Application { tag: "foo" };
        let policy = DataAccessPolicy {
            transforms: vec![
                // This transform won't match because the src index is wrong.
                Transform {
                    src: 0,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                    }),
                    ..Default::default()
                },
                // This transform won't match because the tag is wrong.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some("other".to_owned()),
                    }),
                    ..Default::default()
                },
                // This transform should match.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                    }),
                    ..Default::default()
                },
                // This transform would also match, but the earlier match should take precedence.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            tracker.find_matching_transform(&[], /* node_id=*/ 1, &policy, &app),
            Ok(2)
        );
    }

    #[test]
    fn test_find_matching_transform_without_match() {
        let tracker = BudgetTracker::default();
        let blob_id = "blob-id".as_bytes();
        let policy = DataAccessPolicy {
            transforms: vec![
                Transform {
                    src: 0,
                    application: Some(ApplicationMatcher {
                        tag: Some("tag1".to_owned()),
                    }),
                    ..Default::default()
                },
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some("tag2".to_owned()),
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        // A transform should not be found if the tag doesn't match.
        assert!(tracker
            .find_matching_transform(
                blob_id,
                /* node_id=*/ 1,
                &policy,
                &Application { tag: "no-match" }
            )
            .is_err());
        // A transform should not be found if the index doesn't match.
        assert!(tracker
            .find_matching_transform(
                blob_id,
                /* node_id=*/ 10,
                &policy,
                &Application { tag: "tag1" }
            )
            .is_err());
    }
}
