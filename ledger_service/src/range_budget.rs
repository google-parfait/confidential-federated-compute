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
#[derive(Debug)]
struct RangeBudget {
    // Budgets for specific ranges of BlobId.
    map: RangeMap<BlobId, u32>,
    // Default budget that all entries not covered by any ranges
    // in the map are assumed to have.
    default_budget: u32,
}

fn intersect_ranges(r1: &BlobRange, r2: &BlobRange) -> BlobRange {
    max(r1.start, r2.start)..min(r1.end, r2.end)
}

impl RangeBudget {
    fn new(default_budget: u32) -> Self {
        RangeBudget { map: RangeMap::new(), default_budget }
    }

    fn has_budget(&self, blob_id: &BlobId) -> bool {
        *self.map.get(blob_id).unwrap_or(&self.default_budget) > 0u32
    }

    fn update_budget(&mut self, range: &BlobRange) -> Result<(), micro_rpc::Status> {
        // Collect existing ranges that intersect with the specified range.
        let intersecting_ranges: Vec<_> = self
            .map
            .overlapping(range)
            .map(|(&ref overlapping_range, &budget)| {
                (intersect_ranges(overlapping_range, range), budget)
            })
            .collect();

        // Verify that all existing ranges have a remaining budget.
        if self.default_budget == 0 || intersecting_ranges.iter().any(|(_, budget)| *budget == 0) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "no budget remaining",
            ));
        }

        // Gaps in the range map with no budgets. These are filled with
        // default budget - 1.
        let gaps: Vec<_> = self.map.gaps(range).collect();
        for range in gaps {
            self.map.insert(range.clone(), self.default_budget - 1);
        }

        // Update budget in previously existing ranges.
        for (range, budget) in intersecting_ranges {
            self.map.insert(range, budget - 1);
        }

        Ok(())
    }

    fn to_vec(&self) -> Vec<(BlobRange, u32)> {
        self.map.iter().map(|(r, b)| (r.clone(), *b)).collect()
    }
}

/// PolicyBudget stores budgets for blobs scoped to a single policy
/// and a single public encryption key.
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

    /// Finds the matching policy budget and the first matching transform in the
    /// policy and returns both.
    ///
    /// The `policy_hash` is used as a concise, stable identifier for the
    /// policy; it's the caller's responsibility to ensure that the policy
    /// hash matches the policy.
    ///
    /// If the policy budget doesn't exist, a default one is created and
    /// returned.
    pub fn find_matching_transform<'a>(
        &self,
        node_id: u32,
        policy: &'a DataAccessPolicy,
        policy_hash: &'a [u8],
        app: &'a Application,
        now: Duration,
    ) -> Result<(&'a PolicyBudget, usize), micro_rpc::Status> {
        todo!("not implemented")
    }

    /// Finds the policy budget for the given policy hash.
    pub fn get_policy_budget<'a>(
        &mut self,
        policy_hash: &'a [u8],
    ) -> Result<&'a mut PolicyBudget, micro_rpc::Status> {
        todo!("not implemented")
    }

    /// Explicitly revoke access to a specific blob.
    pub fn revoke(&mut self, blob_id: &BlobId) {
        todo!("not implemented")
    }

    pub fn is_revoked(&self, blob_id: BlobId) {
        todo!("not implemented")
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
    use alloc::{boxed::Box, vec};
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
    fn test_partially_consumed_budget() {
        let mut budget = RangeBudget::new(2);
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.has_budget(&0.into()), true);
        assert_eq!(budget.has_budget(&1.into()), true);
        assert_eq!(budget.has_budget(&2.into()), true);
        assert_eq!(budget.has_budget(&3.into()), true);
        assert_eq!(budget.has_budget(&4.into()), true);
    }

    #[test]
    fn test_fully_consumed_budget() {
        let mut budget = RangeBudget::new(2);
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.has_budget(&0.into()), true);
        assert_eq!(budget.has_budget(&1.into()), false);
        assert_eq!(budget.has_budget(&2.into()), false);
        assert_eq!(budget.has_budget(&3.into()), true);
        assert_eq!(budget.has_budget(&4.into()), true);
    }

    #[test]
    fn test_update_budget_one_range() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4))));
    }

    #[test]
    fn test_update_budget_two_non_overlapping_ranges() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(4, 5)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4)), eq((range(4, 5), 4))));
    }

    #[test]
    fn test_update_budget_two_overlapping_ranges() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 4)), Ok(()));
        assert_eq!(budget.update_budget(&range(3, 5)), Ok(()));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 3), 4)), eq((range(3, 4), 3)), eq((range(4, 5), 4)))
        );
    }

    #[test]
    fn test_update_budget_collapse_adjacent_ranges() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(2, 3)), Ok(()));
        // Should have just one merged range
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 3), 4))));
    }

    #[test]
    fn test_update_budget_collapse_ranges() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(4, 5)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 4)), eq((range(4, 5), 4))));
        // This should fill the gap and merge ranges
        assert_eq!(budget.update_budget(&range(2, 4)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 5), 4))));
    }

    #[test]
    fn test_update_budget_adjacent_ranges_with_different_budget() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(2, 3)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 3)), eq((range(2, 3), 4))));
    }

    #[test]
    fn test_update_budget_over_multiple_ranges() {
        let mut budget = RangeBudget::new(5);
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.update_budget(&range(5, 7)), Ok(()));
        assert_eq!(budget.update_budget(&range(8, 10)), Ok(()));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 3), 3)), eq((range(5, 7), 4)), eq((range(8, 10), 4)))
        );
        // Consume budget over the range that overlaps all ranges and
        // gaps between them.
        assert_eq!(budget.update_budget(&range(2, 9)), Ok(()));
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
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.update_budget(&range(1, 3)), Ok(()));
        assert_eq!(budget.update_budget(&range(4, 6)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 3), 3)), eq((range(4, 6), 4))));
        // Update budget so that it overlaps with the first range
        // and collapses with the second range. This should partially
        // consume the first range, fill the gap and merge with the second
        // range.
        assert_eq!(budget.update_budget(&range(2, 4)), Ok(()));
        assert_that!(
            budget.to_vec(),
            elements_are!(eq((range(1, 2), 3)), eq((range(2, 3), 2)), eq((range(3, 6), 4)))
        );
    }

    #[test]
    fn test_update_budget_to_zero() {
        let mut budget = RangeBudget::new(1);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_that!(budget.to_vec(), elements_are!(eq((range(1, 2), 0))));
    }

    #[test]
    fn test_update_comsumed_budget() {
        let mut budget = RangeBudget::new(1);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_err!(
            budget.update_budget(&range(1, 2)),
            micro_rpc::StatusCode::PermissionDenied,
            "no budget remaining"
        );
    }

    #[test]
    fn test_update_consumed_budget_comlex() {
        let mut budget = RangeBudget::new(2);
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(1, 2)), Ok(()));
        assert_eq!(budget.update_budget(&range(4, 6)), Ok(()));
        // Range (1, 2) is consumed.
        assert_err!(
            budget.update_budget(&range(0, 5)),
            micro_rpc::StatusCode::PermissionDenied,
            "no budget remaining"
        );
    }

    #[test]
    fn test_zero_default_budget() {
        let mut budget = RangeBudget::new(0);
        assert_eq!(budget.has_budget(&1.into()), false);
        assert_err!(
            budget.update_budget(&range(1, 2)),
            micro_rpc::StatusCode::PermissionDenied,
            "no budget remaining"
        );
    }
}
