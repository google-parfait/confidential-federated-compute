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

use crate::blobid::BlobId;

use crate::attestation::Application;
use crate::replication::BudgetSnapshot;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::{
    cmp::{max, min},
    time::Duration,
};
use federated_compute::proto::DataAccessPolicy;
use rangemap::map::RangeMap;

type BlobRange = core::ops::Range<BlobId>;

/// Range budget stores u32 budgets for a single policy node or
/// single shared budget index in a DataAccessPolicy.
/// If there is no entry in the map for any give BlobId, it is
/// assumed that it has a default budget.
#[derive(Debug, Default)]
struct RangeBudget {
    // Budgets for specific ranges of BlobId.
    map: RangeMap<BlobId, u32>,
    // Default budget that all entries not covered by any ranges
    // in the map are assumed to have; None if the budget is unlimited.
    default_budget: Option<u32>,
}

fn intersect_ranges(r1: &BlobRange, r2: &BlobRange) -> BlobRange {
    max(r1.start, r2.start)..min(r1.end, r2.end)
}

impl RangeBudget {
    fn new(default_budget: u32) -> Self {
        RangeBudget { default_budget: Some(default_budget), ..Default::default() }
    }

    fn unlimited() -> Self {
        RangeBudget { default_budget: None, ..Default::default() }
    }

    fn has_budget(&self, blob_id: &BlobId) -> bool {
        if self.default_budget.is_none() {
            // Unlimited budget
            return true;
        }

        self.map.get(blob_id).or(self.default_budget.as_ref()).map(|b| *b > 0).unwrap()
    }

    fn update_budget(&mut self, range: &BlobRange) {
        let default_budget = self.default_budget.unwrap_or(0);
        if default_budget == 0 {
            // Either unlimited or zero budget - either way there is nothing
            // to update.
            return;
        }

        // Collect existing ranges that intersect with the specified range.
        let intersecting_ranges: Vec<_> = self
            .map
            .overlapping(range)
            .map(|(&ref overlapping_range, &budget)| {
                (intersect_ranges(overlapping_range, range), budget)
            })
            .collect();

        // Gaps in the range map with no budgets. These are filled with
        // default budget - 1.
        let gaps: Vec<_> = self.map.gaps(range).collect();
        for range in gaps {
            self.map.insert(range.clone(), default_budget - 1);
        }

        // Update budget in previously existing ranges.
        for (range, budget) in intersecting_ranges {
            if budget > 0 {
                self.map.insert(range, budget - 1);
            }
        }
    }

    fn to_vec(&self) -> Vec<(BlobRange, u32)> {
        self.map.iter().map(|(r, b)| (r.clone(), *b)).collect()
    }
}

/// PolicyBudget stores budgets for blobs scoped to a single policy
/// and a single public encryption key.
#[derive(Default)]
pub struct PolicyBudget {
    // Budgets for specific transform nodes.
    transform_access_budgets: Vec<RangeBudget>,
    // Shared budgets.
    shared_access_budgets: Vec<RangeBudget>,
}

impl PolicyBudget {
    pub fn new(policy: &DataAccessPolicy) -> Self {
        todo!("not implemented")
    }

    /// Checks whether there is a remaining budget for the given blob in the
    /// policy transform at `transform_index` and all corresponding shared
    /// budgets.
    ///
    /// The `policy_hash` is used as a concise, stable identifier for the
    /// policy; it's the caller's responsibility to ensure that the policy
    /// hash matches the policy.
    pub fn has_budget(
        &self,
        blob_id: &BlobId,
        transform_index: usize,
        policy: &DataAccessPolicy,
    ) -> bool {
        todo!("not implemented")
    }

    /// Updates the budget to reflect a new access for all blobs in the range in
    /// the policy transform at `transform_index` and all corresponding
    /// shared budgets.
    ///
    /// Please note that this method never fails. If there wasn't a budget in
    /// the given range or any part of the range, it isn't further reduced below
    /// zero.  It is the caller responsibility to check each blob individually
    /// before updating the budget for the entire range.
    pub fn update_budget(
        &mut self,
        range: &BlobRange,
        transform_index: usize,
        policy: &DataAccessPolicy,
    ) {
        todo!("not implemented")
    }
}

/// A BudgetTracker keeps track of the remaining budgets for zero or more blobs
/// that are scoped to a single public encryption key.
#[derive(Default)]
pub struct BudgetTracker {
    // Per-policy budgets keyed by policy hash.
    budgets: BTreeMap<Vec<u8>, PolicyBudget>,
    // Blob ids whose access has been explicitly revoked.
    revoked_blobs: BTreeSet<BlobId>,
}

/// A BudgetTracker keeps track of the remaining budgets for zero or more blobs
/// ranges.
///
/// The expected usage pattern is the following:
/// 1) Determine the transform_index for the given operation - it must be the
///    same for all blobs in any given range whose budget is being updated.
/// 2) Call `get_policy_budget` once to get the policy for the specific policy
///    hash. It is expected that all blobs in the given range must belong to the
///    same policy.
/// 3) For every blob_id in the range call `is_revoked` on BudgetTracker and
///    `has_budget` on PolicyBudget to check whether that blob can be accessed.
/// 4) Call `update_budget` on PolicyBudget once to record the access.
impl BudgetTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds the first matching transform in the policy and returns its index.
    pub fn find_matching_transform(
        node_id: u32,
        policy: &DataAccessPolicy,
        app: &Application,
        now: Duration,
    ) -> Result<usize, micro_rpc::Status> {
        let mut matched_index: Option<usize> = None;
        for (i, transform) in policy.transforms.iter().enumerate() {
            if transform.src != node_id || !app.matches(&transform.application, now) {
                continue;
            }
            if matched_index.is_some() {
                // Multiple matched transforms.
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::FailedPrecondition,
                    "requesting application matches multiple transforms in the access policy",
                ));
            }

            matched_index = Some(i);
        }

        match matched_index {
            Some(index) => Ok(index),
            None => Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "requesting application does not match the access policy",
            )),
        }
    }

    /// Returns the policy budget for the given policy hash.
    pub fn get_policy_budget<'a>(&'a mut self, policy_hash: &[u8]) -> &'a mut PolicyBudget {
        self.budgets.entry(policy_hash.to_vec()).or_default()
    }

    /// Explicitly revoke access to the blob.
    pub fn revoke(&mut self, blob_id: &BlobId) {
        self.revoked_blobs.insert(blob_id.clone());
    }

    // Check if access to the blob has been revoked.
    pub fn is_revoked(&self, blob_id: &BlobId) -> bool {
        self.revoked_blobs.contains(blob_id)
    }

    /// Saves the entire BudgetTracker state in BudgetSnapshot  as a part of
    /// snapshot replication.
    pub fn save_snapshot(&self) -> BudgetSnapshot {
        todo!("not implemented")
    }

    /// Replaces the entire BudgetTracker state with state loaded from
    /// BudgetSnapshot as a part of snapshot replication.
    pub fn load_snapshot(&mut self, snapshot: BudgetSnapshot) -> Result<(), micro_rpc::Status> {
        todo!("not implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_err;
    use alloc::{borrow::ToOwned, boxed::Box, vec};
    use federated_compute::proto::{ApplicationMatcher, data_access_policy::Transform};
    use googletest::prelude::*;

    fn range(start: u128, end: u128) -> BlobRange {
        start.into()..end.into()
    }

    #[test]
    fn test_default_budget() {
        let budget = RangeBudget::new(1);
        assert_eq!(budget.map.is_empty(), true);
        assert_eq!(budget.has_budget(&1.into()), true);
    }

    #[test]
    fn test_zero_default_budget() {
        let mut budget = RangeBudget::new(0);
        assert_eq!(budget.has_budget(&1.into()), false);
    }

    #[test]
    fn test_unlimited_budget() {
        let budget = RangeBudget::unlimited();
        assert_eq!(budget.map.is_empty(), true);
        assert_eq!(budget.has_budget(&1.into()), true);
    }

    #[test]
    fn test_partially_consumed_budget() {
        let mut budget = RangeBudget::new(2);
        budget.update_budget(&range(1, 3));
        assert_eq!(budget.has_budget(&0.into()), true);
        assert_eq!(budget.has_budget(&1.into()), true);
        assert_eq!(budget.has_budget(&2.into()), true);
        assert_eq!(budget.has_budget(&3.into()), true);
        assert_eq!(budget.has_budget(&4.into()), true);
    }

    #[test]
    fn test_fully_consumed_budget() {
        let mut budget = RangeBudget::new(2);
        budget.update_budget(&range(1, 3));
        budget.update_budget(&range(1, 3));
        assert_eq!(budget.has_budget(&0.into()), true);
        assert_eq!(budget.has_budget(&1.into()), false);
        assert_eq!(budget.has_budget(&2.into()), false);
        assert_eq!(budget.has_budget(&3.into()), true);
        assert_eq!(budget.has_budget(&4.into()), true);
    }

    #[test]
    fn test_update_budget_one_range() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 2));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4))));
    }

    #[test]
    fn test_update_budget_two_non_overlapping_ranges() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(4, 5));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4)), eq((range(4, 5), 4))));
    }

    #[test]
    fn test_update_budget_two_overlapping_ranges() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 4));
        budget.update_budget(&range(3, 5));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 3), 4)), eq((range(3, 4), 3)), eq((range(4, 5), 4)))
        );
    }

    #[test]
    fn test_update_budget_collapse_adjacent_ranges() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(2, 3));
        // Should have just one merged range
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 3), 4))));
    }

    #[test]
    fn test_update_budget_collapse_ranges() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(4, 5));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4)), eq((range(4, 5), 4))));
        // This should fill the gap and merge ranges
        budget.update_budget(&range(2, 4));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 5), 4))));
    }

    #[test]
    fn test_update_budget_adjacent_ranges_with_different_budget() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(2, 3));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 3)), eq((range(2, 3), 4))));
    }

    #[test]
    fn test_update_budget_over_multiple_ranges() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 3));
        budget.update_budget(&range(1, 3));
        budget.update_budget(&range(5, 7));
        budget.update_budget(&range(8, 10));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 3), 3)), eq((range(5, 7), 4)), eq((range(8, 10), 4)))
        );
        // Consume budget over the range that overlaps all ranges and
        // gaps between them.
        budget.update_budget(&range(2, 9));
        assert_that!(
            budget.to_vec(),
            elements_are!(
                eq((range(1, 2), 3)),
                eq((range(2, 3), 2)),
                eq((range(3, 5), 4)),
                eq((range(5, 7), 3)),
                eq((range(7, 8), 4)),
                eq((range(8, 9), 3)),
                eq((range(9, 10), 4))
            )
        );
    }

    #[test]
    fn test_update_budget_complex() {
        let mut budget = RangeBudget::new(5);
        budget.update_budget(&range(1, 3));
        budget.update_budget(&range(1, 3));
        budget.update_budget(&range(4, 6));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 3), 3)), eq((range(4, 6), 4))));
        // Update budget so that it overlaps with the first range
        // and collapses with the second range. This should partially
        // consume the first range, fill the gap and merge with the second
        // range.
        budget.update_budget(&range(2, 4));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 2), 3)), eq((range(2, 3), 2)), eq((range(3, 6), 4)))
        );
    }

    #[test]
    fn test_update_budget_to_zero() {
        let mut budget = RangeBudget::new(1);
        budget.update_budget(&range(1, 2));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 0))));
    }

    #[test]
    fn test_update_comsumed_budget() {
        let mut budget = RangeBudget::new(1);
        budget.update_budget(&range(1, 2));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 0))));
        // Updating the same budget again doesn't change anything.
        budget.update_budget(&range(1, 2));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 0))));
    }

    #[test]
    fn test_update_partially_consumed_budget() {
        let mut budget = RangeBudget::new(2);
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(1, 2));
        budget.update_budget(&range(4, 6));
        // Range (1, 2) is consumed, but that should be OK.
        budget.update_budget(&range(0, 5));
        assert_that!(
            budget.to_vec(),
            elements_are!(
                eq((range(0, 1), 1)),
                eq((range(1, 2), 0)),
                eq((range(2, 4), 1)),
                eq((range(4, 5), 0)),
                eq((range(5, 6), 1)),
            )
        );
    }

    #[test]
    fn test_find_matching_transform_success() {
        let app = Application { tag: "foo", ..Default::default() };
        let policy = DataAccessPolicy {
            transforms: vec![
                // This transform won't match because the src index is wrong.
                Transform {
                    src: 0,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // This transform won't match because the tag is wrong.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some("other".to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // This transform should match.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            BudgetTracker::find_matching_transform(
                /* node_id= */ 0,
                &policy,
                &app,
                Duration::default()
            ),
            Ok(0)
        );
        assert_eq!(
            BudgetTracker::find_matching_transform(
                /* node_id= */ 1,
                &policy,
                &app,
                Duration::default()
            ),
            Ok(2)
        );
    }

    #[test]
    fn test_find_matching_transform_multiple_matches() {
        let app = Application { tag: "foo", ..Default::default() };
        let policy = DataAccessPolicy {
            transforms: vec![
                // This transform won't match because the src index is wrong.
                Transform {
                    src: 0,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // This transform won't match because the tag is wrong.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some("other".to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // This transform should match.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                // This transform would also match, but the earlier match should take precedence.
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some(app.tag.to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        assert_err!(
            BudgetTracker::find_matching_transform(
                /* node_id= */ 1,
                &policy,
                &app,
                Duration::default()
            ),
            micro_rpc::StatusCode::FailedPrecondition,
            "requesting application matches multiple transforms in the access policy"
        );
    }

    #[test]
    fn test_find_matching_transform_no_match() {
        let app = Application { tag: "foo", ..Default::default() };
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 1,
                application: Some(ApplicationMatcher {
                    tag: Some("other".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_err!(
            BudgetTracker::find_matching_transform(
                /* node_id= */ 0,
                &policy,
                &app,
                Duration::default()
            ),
            micro_rpc::StatusCode::FailedPrecondition,
            "requesting application does not match the access policy"
        );
        assert_err!(
            BudgetTracker::find_matching_transform(
                /* node_id= */ 1,
                &policy,
                &app,
                Duration::default()
            ),
            micro_rpc::StatusCode::FailedPrecondition,
            "requesting application does not match the access policy"
        );
    }

    #[test]
    fn test_is_revoked() {
        let mut budget_tracker = BudgetTracker::new();
        budget_tracker.revoke(&1.into());
        assert_eq!(budget_tracker.is_revoked(&1.into()), true);
        assert_eq!(budget_tracker.is_revoked(&2.into()), false);
    }
}
