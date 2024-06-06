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

use crate::attestation::Application;
use crate::replication::{BlobBudgetSnapshot, BudgetSnapshot, PerPolicyBudgetSnapshot};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::time::Duration;
use federated_compute::proto::{
    access_budget::Kind as AccessBudgetKind, AccessBudget, DataAccessPolicy,
};

/// The remaining privacy budget for an individual blob.
#[derive(Default)]
struct BlobBudget {
    transform_access_budgets: Vec<u32>,
    shared_access_budgets: Vec<u32>,
}

impl BlobBudget {
    pub fn new(policy: &DataAccessPolicy) -> Self {
        let mut transform_access_budgets = Vec::with_capacity(policy.transforms.len());
        for transform in &policy.transforms {
            transform_access_budgets.push(match transform.access_budget {
                Some(AccessBudget { kind: Some(AccessBudgetKind::Times(n)), .. }) => n,
                Some(AccessBudget { kind: None }) => 0,
                None => 0,
            })
        }

        let mut shared_access_budgets = Vec::with_capacity(policy.shared_access_budgets.len());
        for access_budget in &policy.shared_access_budgets {
            shared_access_budgets.push(match access_budget.kind {
                Some(AccessBudgetKind::Times(n)) => n,
                None => 0,
            })
        }

        Self { transform_access_budgets, shared_access_budgets }
    }

    /// Returns whether another access is allowed.
    pub fn allows_access(&self, transform_index: usize, policy: &DataAccessPolicy) -> bool {
        let transform = &policy.transforms[transform_index];
        if let Some(ref access_budget) = &transform.access_budget {
            if !Self::has_remaining_budget(
                &self.transform_access_budgets,
                transform_index,
                access_budget,
            ) {
                return false;
            }
        }
        for &shared_index in &transform.shared_access_budget_indices {
            let shared_index = shared_index as usize;
            if shared_index >= policy.shared_access_budgets.len()
                || !Self::has_remaining_budget(
                    &self.shared_access_budgets,
                    shared_index,
                    &policy.shared_access_budgets[shared_index],
                )
            {
                return false;
            }
        }
        true
    }

    /// Returns whether the there's sufficient budget at the specified index for
    /// another access.
    fn has_remaining_budget(budgets: &[u32], index: usize, access_budget: &AccessBudget) -> bool {
        match access_budget.kind {
            Some(AccessBudgetKind::Times(_)) => budgets.get(index).copied().unwrap_or(0) > 0,
            None => true,
        }
    }

    /// Updates the budget to record an access.
    pub fn record_access(
        &mut self,
        transform_index: usize,
        policy: &DataAccessPolicy,
    ) -> Result<(), micro_rpc::Status> {
        let transform = &policy.transforms[transform_index];
        if let Some(ref access_budget) = &transform.access_budget {
            Self::update_remaining_budget(
                &mut self.transform_access_budgets,
                transform_index,
                access_budget,
            )?;
        }
        for &shared_index in &transform.shared_access_budget_indices {
            let shared_index = shared_index as usize;
            let access_budget =
                policy.shared_access_budgets.get(shared_index).ok_or_else(|| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        "AccessPolicy is invalid",
                    )
                })?;
            Self::update_remaining_budget(
                &mut self.shared_access_budgets,
                shared_index,
                access_budget,
            )?;
        }
        Ok(())
    }

    /// Updates the budget with the specified index based on the AccessBudget
    /// type.
    fn update_remaining_budget(
        budgets: &mut [u32],
        index: usize,
        access_budget: &AccessBudget,
    ) -> Result<(), micro_rpc::Status> {
        if let Some(AccessBudgetKind::Times(_)) = access_budget.kind {
            match budgets.get_mut(index) {
                Some(b) if *b > 0 => *b -= 1,
                _ => {
                    return Err(micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::PermissionDenied,
                        "no budget remaining or DataAccessPolicy invalid",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A BudgetTracker keeps track of the remaining budgets for zero or more blobs.
#[derive(Default)]
pub struct BudgetTracker {
    /// Budgets keyed by policy hash and blob id.
    budgets: BTreeMap<Vec<u8>, BTreeMap<Vec<u8>, BlobBudget>>,
    /// Blob ids whose budgets have been consumed.
    consumed_budgets: BTreeSet<Vec<u8>>,
}

impl BudgetTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds the first matching transform in the policy that has sufficient
    /// budget available.
    ///
    /// The `policy_hash` is used as a concise, stable identifier for the
    /// policy; it's the caller's responsibility to ensure that the policy
    /// hash matches the policy.
    pub fn find_matching_transform(
        &self,
        blob_id: &[u8],
        node_id: u32,
        policy: &DataAccessPolicy,
        policy_hash: &[u8],
        app: &Application,
        now: Duration,
    ) -> Result<usize, micro_rpc::Status> {
        if self.consumed_budgets.contains(blob_id) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "data access budget consumed",
            ));
        }

        let mut match_found = false;
        for (i, transform) in policy.transforms.iter().enumerate() {
            if transform.src != node_id || !app.matches(&transform.application, now) {
                continue;
            }
            match_found = true;

            let mut owned_budget = None;
            let budget = self
                .budgets
                .get(policy_hash)
                .and_then(|map| map.get(blob_id))
                .unwrap_or_else(|| owned_budget.insert(BlobBudget::new(policy)));
            if budget.allows_access(i, policy) {
                return Ok(i);
            }
        }

        Err(match match_found {
            true => micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "data access budget exhausted",
            ),
            false => micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "requesting application does not match the access policy",
            ),
        })
    }

    /// Updates the budget for a blob to reflect a new access.
    pub fn update_budget(
        &mut self,
        blob_id: &[u8],
        transform_index: usize,
        policy: &DataAccessPolicy,
        policy_hash: &[u8],
    ) -> Result<(), micro_rpc::Status> {
        if self.consumed_budgets.contains(blob_id) {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                "data access budget consumed",
            ));
        }

        self.budgets
            .entry(policy_hash.to_vec())
            .or_default()
            .entry(blob_id.to_vec())
            .or_insert_with(|| BlobBudget::new(policy))
            .record_access(transform_index, policy)

        // TODO: To reduce memory overhead, consider moving the entry to
        // `consumed_budgets` if the budget has been entirely consumed.
    }

    /// Consumes all remaining budget for a blob, making all future calls to
    /// update_budget fail.
    pub fn consume_budget(&mut self, blob_id: &[u8]) {
        if self.consumed_budgets.insert(blob_id.to_vec()) {
            // If the budget wasn't already consumed, remove any not-yet-consumed budgets
            // since they'll never be accessed.
            for (_, map) in self.budgets.iter_mut() {
                map.remove(blob_id);
            }
        }
    }

    /// Saves the entire BudgetTracker state in BudgetSnapshot  as a part of
    /// snapshot replication.
    pub fn save_snapshot(&self) -> BudgetSnapshot {
        let mut snapshot = BudgetSnapshot::default();

        for (access_policy_sha256, budgets) in &self.budgets {
            let mut per_policy_snapshot = PerPolicyBudgetSnapshot::default();
            per_policy_snapshot.access_policy_sha256 = access_policy_sha256.clone();

            for (blob_id, blob_budget) in budgets {
                per_policy_snapshot.budgets.push(BlobBudgetSnapshot {
                    blob_id: blob_id.clone(),
                    transform_access_budgets: blob_budget.transform_access_budgets.clone(),
                    shared_access_budgets: blob_budget.shared_access_budgets.clone(),
                });
            }

            snapshot.per_policy_snapshots.push(per_policy_snapshot);
        }

        for blob_id in &self.consumed_budgets {
            snapshot.consumed_budgets.push(blob_id.clone());
        }

        snapshot
    }

    /// Replaces the entire BudgetTracker state with state loaded from
    /// BudgetSnapshot as a part of snapshot replication.
    pub fn load_snapshot(&mut self, snapshot: BudgetSnapshot) -> Result<(), micro_rpc::Status> {
        // New state
        let mut new_self = BudgetTracker::new();

        for per_policy_snapshot in snapshot.per_policy_snapshots {
            let mut per_policy_budgets = BTreeMap::<Vec<u8>, BlobBudget>::new();
            for blob_budget_snapshot in per_policy_snapshot.budgets {
                if per_policy_budgets
                    .insert(
                        blob_budget_snapshot.blob_id,
                        BlobBudget {
                            transform_access_budgets: blob_budget_snapshot.transform_access_budgets,
                            shared_access_budgets: blob_budget_snapshot.shared_access_budgets,
                        },
                    )
                    .is_some()
                {
                    return Err(micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        "Duplicated `blob_id` entries in the snapshot",
                    ));
                }
            }
            if new_self
                .budgets
                .insert(per_policy_snapshot.access_policy_sha256, per_policy_budgets)
                .is_some()
            {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "Duplicated `access_policy_sha256` entries in the snapshot",
                ));
            }
        }

        for consumed_blob_id in snapshot.consumed_budgets {
            if !new_self.consumed_budgets.insert(consumed_blob_id) {
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

    use alloc::{borrow::ToOwned, vec};
    use federated_compute::proto::{
        access_budget::Kind as AccessBudgetKind, data_access_policy::Transform, AccessBudget,
        ApplicationMatcher,
    };

    #[test]
    fn test_find_matching_transform_success() {
        let tracker = BudgetTracker::default();
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

        assert_eq!(
            tracker.find_matching_transform(
                &[],
                /* node_id= */ 1,
                &policy,
                b"policy-hash",
                &app,
                Duration::default()
            ),
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
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Transform {
                    src: 1,
                    application: Some(ApplicationMatcher {
                        tag: Some("tag2".to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let policy_hash = b"hash";

        // A transform should not be found if the tag doesn't match.
        assert!(
            tracker
                .find_matching_transform(
                    blob_id,
                    /* node_id= */ 1,
                    &policy,
                    policy_hash,
                    &Application { tag: "no-match", ..Default::default() },
                    Duration::default()
                )
                .is_err()
        );
        // A transform should not be found if the index doesn't match.
        assert!(
            tracker
                .find_matching_transform(
                    blob_id,
                    /* node_id= */ 10,
                    &policy,
                    policy_hash,
                    &Application { tag: "tag1", ..Default::default() },
                    Duration::default()
                )
                .is_err()
        );
    }

    #[test]
    fn test_find_matching_transform_with_invalid_policy() {
        let tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                // An out-of-bounds index should not crash.
                shared_access_budget_indices: vec![10],
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(
            tracker
                .find_matching_transform(
                    &[],
                    /* node_id= */ 0,
                    &policy,
                    b"policy-hash",
                    &Application::default(),
                    Duration::default()
                )
                .is_err()
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
        let policy_hash = b"hash";
        let blob_id = b"blob-id";

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);

        // The remaining budget should now be 1, so the next access should also succeed.
        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);

        // But a third access should fail.
        assert_eq!(
            tracker.find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "data access budget exhausted"
            ))
        );
    }

    #[test]
    fn test_update_budget_after_exhausted() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash = b"hash";
        let blob_id = b"blob-id";

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);

        // There is a small chance that update_budget might be called after there's no
        // remaining budget because of the replication. It should fail in that case.
        assert_eq!(
            tracker.update_budget(blob_id, transform_index, &policy, policy_hash),
            Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "no budget remaining or DataAccessPolicy invalid"
            ))
        );
    }

    #[test]
    fn test_update_budget_after_consume() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform { src: 0, ..Default::default() }],
            ..Default::default()
        };
        let blob_id = b"blob-id";

        tracker.consume_budget(blob_id);

        // update_budget shouldn't be called after consume_budget because
        // find_matching_transforms will have failed. But if it is, it should
        // fail.
        assert_eq!(
            tracker.update_budget(blob_id, /* transform_index= */ 0, &policy, b"policy-hash"),
            Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                "data access budget consumed"
            ))
        );
    }

    #[test]
    fn test_update_budget_with_invalid_policy() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                // An out-of-bounds index should not crash.
                shared_access_budget_indices: vec![10],
                ..Default::default()
            }],
            ..Default::default()
        };

        // update_budget shouldn't be called with an invalid policy because
        // find_matching_transforms will have failed. But if it is, it should
        // fail.
        assert!(
            tracker
                .update_budget(b"blob-id", /* transform_index= */ 0, &policy, b"policy-hash")
                .is_err()
        );
    }

    #[test]
    fn test_consume_budget() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform { src: 0, ..Default::default() }],
            ..Default::default()
        };
        let policy_hash = b"hash";
        let blob_id = b"blob-id";

        tracker.consume_budget(blob_id);

        assert_eq!(
            tracker.find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "data access budget consumed"
            ))
        );

        // Access should still be allowed for a different blob.
        assert_eq!(
            tracker.find_matching_transform(
                b"blob-id2",
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Ok(0)
        );
    }

    #[test]
    fn test_shared_budgets() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![
                Transform {
                    src: 0,
                    access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                    shared_access_budget_indices: vec![0],
                    ..Default::default()
                },
                Transform { src: 0, shared_access_budget_indices: vec![0], ..Default::default() },
            ],
            shared_access_budgets: vec![AccessBudget { kind: Some(AccessBudgetKind::Times(2)) }],
            ..Default::default()
        };
        let policy_hash = b"hash";
        let blob_id = b"blob-id";

        // The first request for access should match the first transform.
        assert_eq!(
            tracker.find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Ok(0)
        );
        assert_eq!(
            tracker.update_budget(blob_id, /* transform_index= */ 0, &policy, policy_hash),
            Ok(())
        );

        // The second should match the second transform since the first's budget is
        // exhausted.
        assert_eq!(
            tracker.find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Ok(1)
        );
        assert_eq!(
            tracker.update_budget(blob_id, /* transform_index= */ 1, &policy, policy_hash),
            Ok(())
        );

        // The third request should fail because the shared budget has now been
        // exhausted.
        assert_eq!(
            tracker.find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::PermissionDenied,
                "data access budget exhausted",
            ))
        );

        // A request for a different blob id (but the same node id) should succeed since
        // budgets are tracked per blob id.
        assert_eq!(
            tracker.find_matching_transform(
                b"blob-id2",
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default()
            ),
            Ok(0)
        );
    }

    #[test]
    fn test_policy_isolation() {
        let mut tracker = BudgetTracker::default();
        let app = Application { tag: "tag", ..Default::default() };
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
        let policy_hash1 = b"hash1";
        let policy2 = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                application: Some(ApplicationMatcher {
                    tag: Some(app.tag.to_owned()),
                    ..Default::default()
                }),
                access_budget: Some(AccessBudget {
                    kind: Some(AccessBudgetKind::Times(1)),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash2 = b"hash2";
        let blob_id = b"blob-id";

        // Budgets for different policies should be tracked separately -- especially to
        // prevent malicious blob id collisions from causing incorrect tracking.
        // If the budgets were shared, the second access would fail.
        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy1,
                policy_hash1,
                &app,
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy1, policy_hash1), Ok(()));

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy2,
                policy_hash2,
                &app,
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy2, policy_hash2), Ok(()));
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
        let blob_id = b"blob-id";

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);

        assert_eq!(
            tracker.save_snapshot(),
            BudgetSnapshot {
                per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                    access_policy_sha256: policy_hash.to_vec(),
                    budgets: vec![BlobBudgetSnapshot {
                        blob_id: blob_id.to_vec(),
                        transform_access_budgets: vec![1],
                        shared_access_budgets: vec![],
                    }],
                }],
                consumed_budgets: vec![],
            }
        );
    }

    #[test]
    fn test_consumed_budget_snapshot() {
        let mut tracker = BudgetTracker::default();
        let policy = DataAccessPolicy {
            transforms: vec![Transform {
                src: 0,
                access_budget: Some(AccessBudget { kind: Some(AccessBudgetKind::Times(1)) }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let policy_hash = b"hash";
        let blob_id = b"blob-id";

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);
        tracker.consume_budget(blob_id);

        assert_eq!(
            tracker.save_snapshot(),
            BudgetSnapshot {
                per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                    access_policy_sha256: policy_hash.to_vec(),
                    budgets: vec![],
                }],
                consumed_budgets: vec![blob_id.to_vec()],
            }
        );
    }

    #[test]
    fn test_load_snapshot() {
        let mut tracker = BudgetTracker::default();

        let snapshot = BudgetSnapshot {
            per_policy_snapshots: vec![
                PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash1".to_vec(),
                    budgets: vec![BlobBudgetSnapshot {
                        blob_id: b"blob1".to_vec(),
                        transform_access_budgets: vec![1],
                        shared_access_budgets: vec![],
                    }],
                },
                PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash2".to_vec(),
                    budgets: vec![
                        BlobBudgetSnapshot {
                            blob_id: b"blob2".to_vec(),
                            transform_access_budgets: vec![2, 3],
                            shared_access_budgets: vec![11],
                        },
                        BlobBudgetSnapshot {
                            blob_id: b"blob3".to_vec(),
                            transform_access_budgets: vec![],
                            shared_access_budgets: vec![12, 13, 14],
                        },
                    ],
                },
            ],
            consumed_budgets: vec![b"blob4".to_vec(), b"blob5".to_vec()],
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
        let blob_id = b"blob-id";

        let transform_index = tracker
            .find_matching_transform(
                blob_id,
                /* node_id= */ 0,
                &policy,
                policy_hash,
                &Application::default(),
                Duration::default(),
            )
            .unwrap();
        assert_eq!(tracker.update_budget(blob_id, transform_index, &policy, policy_hash), Ok(()),);
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
                        budgets: vec![],
                    },
                    PerPolicyBudgetSnapshot {
                        access_policy_sha256: b"hash1".to_vec(),
                        budgets: vec![],
                    }
                ],
                consumed_budgets: vec![]
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "Duplicated `access_policy_sha256` entries in the snapshot"
        );
    }

    #[test]
    fn test_load_snapshot_duplicated_blob_id() {
        let mut tracker = BudgetTracker::default();
        assert_err!(
            tracker.load_snapshot(BudgetSnapshot {
                per_policy_snapshots: vec![PerPolicyBudgetSnapshot {
                    access_policy_sha256: b"hash1".to_vec(),
                    budgets: vec![
                        BlobBudgetSnapshot { blob_id: b"blob1".to_vec(), ..Default::default() },
                        BlobBudgetSnapshot { blob_id: b"blob1".to_vec(), ..Default::default() },
                    ],
                },],
                consumed_budgets: vec![]
            }),
            micro_rpc::StatusCode::InvalidArgument,
            "Duplicated `blob_id` entries in the snapshot"
        );
    }

    #[test]
    fn test_load_snapshot_duplicated_consumed_blob() {
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
}
