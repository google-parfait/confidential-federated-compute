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

use crate::attestation::Application;
use crate::blobid::BlobId;
use crate::replication::{BudgetSnapshot, PerPolicyBudgetSnapshot, RangeBudgetSnapshot};

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format, vec,
    vec::Vec,
};
use core::{
    cmp::{max, min},
    time::Duration,
};
use federated_compute::proto::{
    AccessBudget, DataAccessPolicy, access_budget::Kind as AccessBudgetKind,
};
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

    fn save_snapshot(&self) -> RangeBudgetSnapshot {
        let mut range_budget_snapshot = RangeBudgetSnapshot::default();
        range_budget_snapshot.default_budget = self.default_budget;

        for (range, budget) in self.map.iter() {
            range_budget_snapshot.start.push(range.start.to_vec());
            range_budget_snapshot.end.push(range.end.to_vec());
            range_budget_snapshot.remaining_budget.push(*budget);
        }
        range_budget_snapshot
    }

    fn load_snapshot(
        range_budget_snapshot: RangeBudgetSnapshot,
    ) -> Result<RangeBudget, micro_rpc::Status> {
        let all_lengths_equal = range_budget_snapshot.start.len()
            == range_budget_snapshot.end.len()
            && range_budget_snapshot.start.len() == range_budget_snapshot.remaining_budget.len();
        if !all_lengths_equal {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "RangeBudgetSnapshot.start len() must be equal to RangeBudgetSnapshot.end and RangeBudgetSnapshot.remaining_budget len().",
            ));
        }

        if range_budget_snapshot.default_budget.is_none() {
            // Unlimited budget isn't supposed to have any entries, so return straightaway.
            return Ok(RangeBudget::unlimited());
        }

        let mut range_budget = RangeBudget::new(range_budget_snapshot.default_budget.unwrap());
        for i in 0..range_budget_snapshot.start.len() {
            let start_blob_id = BlobId::from_vec(&range_budget_snapshot.start[i]).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!(
                        "Invalid `blob_id` in the range_budget_snapshot.start for index: {:?} err:{:?}",
                        i, err
                    ),
                )
            })?;

            let end_blob_id = BlobId::from_vec(&range_budget_snapshot.end[i]).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!(
                        "Invalid `blob_id` in the range_budget_snapshot.end for index: {:?} err:{:?}",
                        i, err
                    ),
                )
            })?;

            range_budget.map.insert(
                BlobRange { start: start_blob_id, end: end_blob_id },
                range_budget_snapshot.remaining_budget[i],
            );
        }
        Ok(range_budget)
    }
}

/// PolicyBudget stores budgets for blobs scoped to a single policy
/// and a single public encryption key.
#[derive(Debug)]
struct PolicyBudget {
    // Budgets for specific transform nodes.
    transform_access_budgets: Vec<RangeBudget>,
    // Shared budgets.
    shared_access_budgets: Vec<RangeBudget>,
}

impl PolicyBudget {
    fn new(policy: &DataAccessPolicy) -> Self {
        let mut transform_access_budgets = Vec::with_capacity(policy.transforms.len());
        for transform in &policy.transforms {
            transform_access_budgets.push(match transform.access_budget {
                Some(AccessBudget { kind: Some(AccessBudgetKind::Times(n)), .. }) => {
                    RangeBudget::new(n)
                }
                Some(AccessBudget { kind: None }) => RangeBudget::new(0),
                None => RangeBudget::unlimited(),
            });
        }

        let mut shared_access_budgets = Vec::with_capacity(policy.shared_access_budgets.len());
        for access_budget in &policy.shared_access_budgets {
            shared_access_budgets.push(match access_budget.kind {
                Some(AccessBudgetKind::Times(n)) => RangeBudget::new(n),
                None => RangeBudget::new(0),
            })
        }
        PolicyBudget { transform_access_budgets, shared_access_budgets }
    }

    /// Checks whether there is a remaining budget for the given blob in the
    /// policy transform at `transform_index` and all corresponding shared
    /// budgets.
    fn has_budget(
        &self,
        blob_id: &BlobId,
        transform_index: usize,
        shared_access_budget_indices: &Vec<u32>,
    ) -> bool {
        // Check if the budget is available for this specific transform.
        if !self.transform_access_budgets[transform_index].has_budget(blob_id) {
            return false;
        }

        // Check corresponding shared budgets.
        for &shared_index in shared_access_budget_indices {
            let shared_index = shared_index as usize;
            if shared_index >= self.shared_access_budgets.len()
                || !self.shared_access_budgets[shared_index].has_budget(blob_id)
            {
                return false;
            }
        }
        true
    }

    /// Updates the budget to reflect a new access for all blobs in the range in
    /// the policy transform at `transform_index` and all corresponding
    /// shared budgets.
    ///
    /// Please note that this method never fails. If there wasn't a budget in
    /// the given range or any part of the range, it isn't further reduced below
    /// zero.  It is the caller responsibility to check each blob individually
    /// before updating the budget for the entire range.
    fn update_budget(
        &mut self,
        range: &BlobRange,
        transform_index: usize,
        shared_access_budget_indices: &Vec<u32>,
    ) {
        // Update the specific transform budget.
        self.transform_access_budgets[transform_index].update_budget(range);

        // Update corresponding shared budgets.
        for &shared_index in shared_access_budget_indices {
            let shared_index = shared_index as usize;
            if shared_index < self.shared_access_budgets.len() {
                self.shared_access_budgets[shared_index].update_budget(range);
            }
        }
    }
}

/// PolicyBudgetTracker is designed to be used in a scope of a single operation
/// that tests and updates multiple blobs that match the same policy and the
/// same transform.
/// The main purpose of PolicyBudgetTracker is bundle multiple parameters
/// together and simplify invocation of `has_budget` and `update_budget` on a
/// policy budget.
#[derive(Debug)]
pub struct PolicyBudgetTracker<'a> {
    policy_budget: &'a mut PolicyBudget,
    transform_index: usize,
    shared_access_budget_indices: &'a Vec<u32>,
    revoked_blobs: &'a BTreeSet<BlobId>,
}

impl PolicyBudgetTracker<'_> {
    /// Checks whether there is a remaining budget for the given blob.
    pub fn has_budget(&self, blob_id: &BlobId) -> bool {
        // Check if access to the blob has been explicitly revoked.
        if self.revoked_blobs.contains(blob_id) {
            return false;
        }
        self.policy_budget.has_budget(
            blob_id,
            self.transform_index,
            self.shared_access_budget_indices,
        )
    }

    /// Updates the budget to reflect a new access for all blobs in the range.
    pub fn update_budget(&mut self, range: &BlobRange) {
        self.policy_budget.update_budget(
            range,
            self.transform_index,
            self.shared_access_budget_indices,
        )
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
/// 1) Determine the `transform_index for the given operation - it must be the
///    same for all blobs in any given range whose budget is being updated.
/// 2) Call `get_policy_budget` once to get the PolicyBudgetTracker for the
///    specific context, including the policy hash and `transform_index. It is
///    expected that all blobs in the given range must belong to the same policy
///    and the same transform.
/// 3) For every blob_id in the range call `has_budget` on PolicyBudgetTracker
///    to check whether access to that blob can be authorized.
/// 4) Call `update_budget` on PolicyBudgetTracker once to record the access.
impl BudgetTracker {
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

    /// Returns the policy budget tracker for the given policy hash and the
    /// transform index.
    pub fn get_policy_budget<'a>(
        &'a mut self,
        policy_hash: &[u8],
        policy: &'a DataAccessPolicy,
        transform_index: usize,
    ) -> Result<PolicyBudgetTracker<'a>, micro_rpc::Status> {
        if transform_index >= policy.transforms.len() {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "transform index does not match the access policy",
            ));
        }
        let shared_access_budget_indices =
            &policy.transforms[transform_index].shared_access_budget_indices;
        for &shared_index in shared_access_budget_indices {
            let shared_index = shared_index as usize;
            if shared_index >= policy.shared_access_budgets.len() {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::FailedPrecondition,
                    "access policy has invalid shared budgets",
                ));
            }
        }
        let policy_budget =
            self.budgets.entry(policy_hash.to_vec()).or_insert_with(|| PolicyBudget::new(policy));
        Ok(PolicyBudgetTracker {
            policy_budget,
            transform_index,
            shared_access_budget_indices,
            revoked_blobs: &self.revoked_blobs,
        })
    }

    /// Explicitly revoke access to the blob.
    pub fn revoke(&mut self, blob_id: &BlobId) {
        self.revoked_blobs.insert(blob_id.clone());
    }

    /// Saves the entire BudgetTracker state in BudgetSnapshot  as a part of
    /// snapshot replication.
    pub fn save_snapshot(&self) -> BudgetSnapshot {
        let mut snapshot = BudgetSnapshot::default();

        for (access_policy_sha256, policy_budget) in &self.budgets {
            let mut per_policy_snapshot = PerPolicyBudgetSnapshot::default();
            per_policy_snapshot.access_policy_sha256 = access_policy_sha256.clone();

            for range_budget in &policy_budget.transform_access_budgets {
                per_policy_snapshot.transform_access_budgets.push(range_budget.save_snapshot());
            }
            for range_budget in &policy_budget.shared_access_budgets {
                per_policy_snapshot.shared_access_budgets.push(range_budget.save_snapshot());
            }
            snapshot.per_policy_snapshots.push(per_policy_snapshot);
        }

        for blob_id in &self.revoked_blobs {
            snapshot.consumed_budgets.push(blob_id.to_vec());
        }

        snapshot
    }

    /// Replaces the entire BudgetTracker state with state loaded from
    /// BudgetSnapshot as a part of snapshot replication.
    pub fn load_snapshot(&mut self, snapshot: BudgetSnapshot) -> Result<(), micro_rpc::Status> {
        let mut new_self = BudgetTracker::default();

        for per_policy_snapshot in snapshot.per_policy_snapshots {
            let mut transform_access_budgets = vec![];
            let mut shared_access_budgets = vec![];
            for range_budget_snapshot in per_policy_snapshot.transform_access_budgets {
                transform_access_budgets.push(RangeBudget::load_snapshot(range_budget_snapshot)?);
            }
            for range_budget_snapshot in per_policy_snapshot.shared_access_budgets {
                shared_access_budgets.push(RangeBudget::load_snapshot(range_budget_snapshot)?);
            }

            let policy_budget = PolicyBudget { transform_access_budgets, shared_access_budgets };
            if new_self
                .budgets
                .insert(per_policy_snapshot.access_policy_sha256, policy_budget)
                .is_some()
            {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "Duplicated `access_policy_sha256` entries in the snapshot",
                ));
            }
        }

        for consumed_blob_id in snapshot.consumed_budgets {
            let revoked_blob_id = BlobId::from_vec(&consumed_blob_id).map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("Invalid `blob_id` in the snapshot: {:?}", err),
                )
            })?;
            if !new_self.revoked_blobs.insert(revoked_blob_id) {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "Duplicated `consumed_budgets` entries in the snapshot",
                ));
            }
        }

        // Replace the state with the new state.
        *self = new_self;
        Ok(())
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
        let budget = RangeBudget::new(0);
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
    fn test_get_policy_budget_with_invalid_transform_index() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_err!(
            tracker.get_policy_budget(b"policy_hash", &policy, /* transform_index */ 1),
            micro_rpc::StatusCode::FailedPrecondition,
            "transform index does not match the access policy"
        );
    }

    #[test]
    fn test_get_policy_budget_with_invalid_shared_budgets() {
        let mut tracker = BudgetTracker::default();
        // The second Transform has invalid shared_access_budget_indices.
        let policy = DataAccessPolicy {
            transforms: vec![
                Transform { src: 0, shared_access_budget_indices: vec![0], ..Default::default() },
                Transform {
                    src: 1,
                    shared_access_budget_indices: vec![0, 1],
                    ..Default::default()
                },
            ],
            shared_access_budgets: vec![AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }],
            ..Default::default()
        };
        assert_err!(
            tracker.get_policy_budget(b"policy_hash", &policy, /* transform_index */ 1),
            micro_rpc::StatusCode::FailedPrecondition,
            "access policy has invalid shared budgets"
        );
    }

    #[test]
    fn test_update_budget() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let transform_index = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget =
            tracker.get_policy_budget(b"policy_hash", &policy, transform_index).unwrap();
        assert_eq!(policy_budget.has_budget(&1.into()), true);
        // Update range [0,5)
        policy_budget.update_budget(&range(0, 5));
        // Verify that there is still budget in that range
        assert_eq!(policy_budget.has_budget(&3.into()), true);
        // Update range [0,5) again
        policy_budget.update_budget(&range(0, 5));
        // Now there shouldn't be a remaining budget in that range but still available
        // budget outside the range.
        assert_eq!(policy_budget.has_budget(&2.into()), false);
        assert_eq!(policy_budget.has_budget(&6.into()), true);
        // Updating range [0, 5) again is fine even though there isn't any remaining
        // budget.
        policy_budget.update_budget(&range(0, 5));
        assert_eq!(policy_budget.has_budget(&2.into()), false);
        assert_eq!(policy_budget.has_budget(&6.into()), true);
    }

    #[test]
    fn test_update_budget_after_revoked() {
        let mut tracker = BudgetTracker::default();
        // Revoke access to one blob.
        tracker.revoke(&3.into());

        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let transform_index = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget =
            tracker.get_policy_budget(b"policy_hash", &policy, transform_index).unwrap();
        // Update range [0, 5)
        policy_budget.update_budget(&range(0, 5));
        // Access to the revoked blob is no longer possible but other blobs in the same
        // range are OK.
        assert_eq!(policy_budget.has_budget(&3.into()), false);
        assert_eq!(policy_budget.has_budget(&2.into()), true);
    }

    #[test]
    fn test_shared_budgets() {
        let mut tracker = BudgetTracker::default();
        // Two transforms share the same shared budget.
        let policy = DataAccessPolicy {
            transforms: vec![
                Transform { src: 0, shared_access_budget_indices: vec![0], ..Default::default() },
                Transform { src: 1, shared_access_budget_indices: vec![0], ..Default::default() },
            ],
            shared_access_budgets: vec![AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }],
            ..Default::default()
        };
        let transform_index1 = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        assert_eq!(transform_index1, 0);
        let mut policy_budget1 =
            tracker.get_policy_budget(b"policy_hash", &policy, transform_index1).unwrap();
        assert_eq!(policy_budget1.has_budget(&2.into()), true);
        policy_budget1.update_budget(&range(0, 5));

        let transform_index2 = BudgetTracker::find_matching_transform(
            /* node_id= */ 1,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        assert_eq!(transform_index2, 1);
        let policy_budget2 =
            tracker.get_policy_budget(b"policy_hash", &policy, transform_index2).unwrap();
        // The range [0, 5) should already be consumed via the shared budget but it
        // should be OK outside of that range.
        assert_eq!(policy_budget2.has_budget(&3.into()), false);
        assert_eq!(policy_budget2.has_budget(&6.into()), true);
    }

    #[test]
    fn test_policy_isolation() {
        let mut tracker = BudgetTracker::default();
        let policy1 = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget {
                    kind: Some(AccessBudgetKind::Times(1)),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let transform_index1 = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy1,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget1 =
            tracker.get_policy_budget(b"policy_hash1", &policy1, transform_index1).unwrap();
        // Update budget for the range [0, 5) and verify that there is no access.
        policy_budget1.update_budget(&range(0, 5));
        assert_eq!(policy_budget1.has_budget(&3.into()), false);

        let policy2 = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget {
                    kind: Some(AccessBudgetKind::Times(1)),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let transform_index2 = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy2,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let policy_budget2 =
            tracker.get_policy_budget(b"policy_hash2", &policy2, transform_index2).unwrap();
        // Verify that there is budget for a blob in the same range [0, 5)
        assert_eq!(policy_budget2.has_budget(&3.into()), true);
    }

    #[test]
    fn test_updated_budget_snapshot() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash = b"hash";

        let transform_index = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget =
            tracker.get_policy_budget(policy_hash, &policy, transform_index).unwrap();
        let start: BlobId = 0.into();
        let end: BlobId = 5.into();
        policy_budget.update_budget(&BlobRange { start, end });

        assert_eq!(tracker.save_snapshot(), BudgetSnapshot {
            per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                access_policy_sha256: policy_hash.to_vec(),
                transform_access_budgets: vec![RangeBudgetSnapshot {
                    start: vec![start.to_vec()],
                    end: vec![end.to_vec()],
                    remaining_budget: vec![1],
                    default_budget: Some(2),
                }],
                ..Default::default()
            }],
            consumed_budgets: vec![],
        });
    }

    #[test]
    fn test_revoked_budget_snapshot() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash = b"hash";

        let transform_index = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget =
            tracker.get_policy_budget(policy_hash, &policy, transform_index).unwrap();
        let start: BlobId = 0.into();
        let end: BlobId = 5.into();
        policy_budget.update_budget(&BlobRange { start, end });
        let revoked_blob_id: BlobId = 3.into();
        tracker.revoke(&3.into());

        assert_eq!(tracker.save_snapshot(), BudgetSnapshot {
            per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                access_policy_sha256: policy_hash.to_vec(),
                transform_access_budgets: vec![RangeBudgetSnapshot {
                    start: vec![start.to_vec()],
                    end: vec![end.to_vec()],
                    remaining_budget: vec![1],
                    default_budget: Some(2),
                }],
                ..Default::default()
            }],
            consumed_budgets: vec![revoked_blob_id.to_vec()],
        });
    }

    #[test]
    fn test_load_snapshot() {
        // Blob IDs have to be 16 byte long in this test to avoid failing the test
        // due to zero padding when saving the snapshot.
        let mut tracker = BudgetTracker::default();

        let snapshot = BudgetSnapshot {
            per_policy_snapshots: vec![
                PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash1".to_vec(),
                    transform_access_budgets: vec![RangeBudgetSnapshot {
                        start: vec![b"_____blob_____1_".to_vec()],
                        end: vec![b"_____blob_____2_".to_vec()],
                        remaining_budget: vec![1],
                        default_budget: Some(2),
                    }],
                    ..Default::default()
                },
                PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash2".to_vec(),
                    transform_access_budgets: vec![
                        RangeBudgetSnapshot {
                            start: vec![b"_____blob_____2_".to_vec()],
                            end: vec![b"_____blob_____3_".to_vec()],
                            remaining_budget: vec![2],
                            default_budget: Some(50),
                        },
                        RangeBudgetSnapshot {
                            start: vec![b"_____blob_____2_".to_vec()],
                            end: vec![b"_____blob_____3_".to_vec()],
                            remaining_budget: vec![3],
                            default_budget: Some(50),
                        },
                    ],
                    shared_access_budgets: vec![
                        RangeBudgetSnapshot {
                            start: vec![b"_____blob_____2_".to_vec(), b"_____blob_____3_".to_vec()],
                            end: vec![b"_____blob_____3_".to_vec(), b"_____blob_____4_".to_vec()],
                            remaining_budget: vec![11, 12],
                            default_budget: Some(50),
                        },
                        RangeBudgetSnapshot {
                            start: vec![b"_____blob_____3_".to_vec()],
                            end: vec![b"_____blob_____4_".to_vec()],
                            remaining_budget: vec![13],
                            default_budget: Some(50),
                        },
                        RangeBudgetSnapshot {
                            start: vec![b"_____blob_____3_".to_vec()],
                            end: vec![b"_____blob_____4_".to_vec()],
                            remaining_budget: vec![14],
                            default_budget: Some(50),
                        },
                    ],
                    ..Default::default()
                },
            ],
            consumed_budgets: vec![b"_____blob_____4_".to_vec(), b"_____blob_____5_".to_vec()],
        };

        // Load the snapshot.
        assert_eq!(tracker.load_snapshot(snapshot.clone()), Ok(()));
        // Save the new snapshot and verify that it is the same as the loaded one.
        assert_eq!(tracker.save_snapshot(), snapshot);
    }

    #[test]
    fn test_load_snapshot_replaces_state() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash = b"hash";

        let transform_index = BudgetTracker::find_matching_transform(
            /* node_id= */ 0,
            &policy,
            &Application::default(),
            Duration::default(),
        )
        .unwrap();
        let mut policy_budget =
            tracker.get_policy_budget(policy_hash, &policy, transform_index).unwrap();
        policy_budget.update_budget(&range(0, 5));
        assert_ne!(tracker.save_snapshot(), BudgetSnapshot::default());

        // Load an empty snapshot and verify that an empty snapshot is saved.
        assert_eq!(tracker.load_snapshot(BudgetSnapshot::default()), Ok(()));
        assert_eq!(tracker.save_snapshot(), BudgetSnapshot::default());
    }

    #[test]
    fn test_load_snapshot_duplicated_policy_hash() {
        let mut tracker = BudgetTracker::default();
        assert_err!(
            tracker.load_snapshot(BudgetSnapshot {
                per_policy_snapshots: vec![
                    PerPolicyBudgetSnapshot {
                        access_policy_sha256: b"hash1".to_vec(),
                        ..Default::default()
                    },
                    PerPolicyBudgetSnapshot {
                        access_policy_sha256: b"hash1".to_vec(),
                        ..Default::default()
                    }
                ],
                consumed_budgets: vec![]
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "Duplicated `access_policy_sha256` entries in the snapshot"
        );
    }

    #[test]
    fn test_load_snapshot_duplicated_revoked_blob() {
        let mut tracker = BudgetTracker::default();
        assert_err!(
            tracker.load_snapshot(BudgetSnapshot {
                consumed_budgets: vec![b"blob1".to_vec(), b"blob1".to_vec()],
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "Duplicated `consumed_budgets` entries in the snapshot"
        );
    }

    #[test]
    fn test_load_snapshot_invalid_range() {
        let mut tracker = BudgetTracker::default();
        assert_err!(
            tracker.load_snapshot(BudgetSnapshot {
                per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash1".to_vec(),
                    transform_access_budgets: vec![RangeBudgetSnapshot {
                        start: vec![b"_____blob_____1_".to_vec(), b"_____blob_____2_".to_vec()],
                        end: vec![b"_____blob_____2_".to_vec()],
                        remaining_budget: vec![1],
                        default_budget: Some(2),
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "RangeBudgetSnapshot.start len() must be equal to RangeBudgetSnapshot.end and RangeBudgetSnapshot.remaining_budget len()."
        );
    }
}
