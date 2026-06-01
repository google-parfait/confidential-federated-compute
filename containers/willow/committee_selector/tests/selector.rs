// Copyright 2026 The Trusted Computations Platform Authors.
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

extern crate willow_committee_selector_service;

mod test {
    use slog::{o, Discard, Logger};
    use willow_committee_selector_service::apps::willow::committee_selector::service::{
        CommitteeSelectorSnapshot, CommitteeSnapshot, CommitteeStatus, EndorsementStatus,
        VolunteerKeyDigest,
    };
    use willow_committee_selector_service::selector::{
        fingerprint, BloomFilter, CommitteeSelector,
    };

    #[test]
    fn test_selector_create_committee() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);

        assert!(selector.create_committee(1, &logger));
        assert!(selector.create_committee(2, &logger));

        // Should fail to create duplicate
        assert!(!selector.create_committee(1, &logger));

        // Should remove oldest (1) and add 3
        assert!(selector.create_committee(3, &logger));

        assert!(selector.check_committee_status(1).is_none());
        assert!(selector.check_committee_status(2).is_some());
        assert!(selector.check_committee_status(3).is_some());
    }

    #[test]
    fn test_volunteer_batch() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);
        assert!(selector.create_committee(1, &logger));

        // Batch with valid and invalid volunteers
        let volunteers = vec![
            VolunteerKeyDigest {
                public_key_digest: 101,
                key_endorsement_status: EndorsementStatus::Valid as i32,
            },
            VolunteerKeyDigest {
                public_key_digest: 102,
                key_endorsement_status: EndorsementStatus::Invalid as i32,
            },
            VolunteerKeyDigest {
                public_key_digest: 103,
                key_endorsement_status: EndorsementStatus::Valid as i32,
            },
        ];

        let assignments = selector.volunteer_batch(volunteers, 42);

        // There should be 3 assignment records returned, but only valid ones assigned
        // to committee 1
        assert_eq!(assignments.len(), 3);

        // 101 is valid -> assigned to committee 1
        assert_eq!(assignments[0].member_public_key_digest, 101);
        assert_eq!(assignments[0].committee_id, 1);

        // 102 is invalid -> committee_id is -1
        assert_eq!(assignments[1].member_public_key_digest, 102);
        assert_eq!(assignments[1].committee_id, -1);

        // 103 is valid -> assigned to committee 1
        assert_eq!(assignments[2].member_public_key_digest, 103);
        assert_eq!(assignments[2].committee_id, 1);

        // Check committee status members
        let status = selector.check_committee_status(1).unwrap();
        assert_eq!(status.0, CommitteeStatus::AcceptingVolunteers);
        assert_eq!(status.1, vec![101, 103]);
        assert_eq!(status.2, 0);
    }

    #[test]
    fn test_next_committee_assignment() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(3);
        assert!(selector.create_committee(1, &logger));
        assert!(selector.create_committee(2, &logger));

        // Eligible committees: [1, 2] (sorted order)
        // Even randomness should return committee 1 (index 0)
        assert_eq!(selector.next_committee_assignment(0), Some(1));
        assert_eq!(selector.next_committee_assignment(2), Some(1));

        // Odd randomness should return committee 2 (index 1)
        assert_eq!(selector.next_committee_assignment(1), Some(2));
        assert_eq!(selector.next_committee_assignment(3), Some(2));
    }

    #[test]
    fn test_sample_committee() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);
        assert!(selector.create_committee(1, &logger));

        // Sampling with not enough volunteers should fail
        assert!(selector.sample_committee(1).is_err());

        // Sampling non-existent committee should fail
        assert!(selector.sample_committee(99).is_err());

        // Use load_snapshot to set up a committee with enough volunteers
        let snapshot = CommitteeSelectorSnapshot {
            committees: vec![CommitteeSnapshot {
                committee_id: 1,
                status: CommitteeStatus::AcceptingVolunteers as i32,
                sequential_order: 1,
                member_public_key_digests: vec![101, 102],
                total_volunteers_seen: 1024 * 1024, // MIN_NUMBER_OF_VOLUNTEERS_PER_COMMITTEE
                bloom_filter: vec![0u8; 2 * 1024 * 1024],
                rejected_volunteers_count: 0,
            }],
        };

        assert!(selector.load_snapshot(snapshot).is_ok());

        // Now sample should succeed
        let result = selector.sample_committee(1);
        assert!(result.is_ok());
        let (status, members, rejected_count) = result.unwrap();
        assert_eq!(status, CommitteeStatus::SelectionComplete);
        assert_eq!(members, vec![101, 102]);
        assert_eq!(rejected_count, 0);

        // Sampling a completed committee should fail
        assert!(selector.sample_committee(1).is_err());
    }

    #[test]
    fn test_remove_oldest_committee() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(3);
        assert!(selector.create_committee(1, &logger));
        assert!(selector.create_committee(2, &logger));

        // Directly call remove_oldest_committee
        selector.remove_oldest_committee(&logger);

        // Committee 1 should be removed as it has the lowest sequential order
        assert!(selector.check_committee_status(1).is_none());
        assert!(selector.check_committee_status(2).is_some());
    }

    #[test]
    fn test_snapshot_save_and_load() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);
        assert!(selector.create_committee(1, &logger));

        let volunteers = vec![VolunteerKeyDigest {
            public_key_digest: 123,
            key_endorsement_status: EndorsementStatus::Valid as i32,
        }];
        selector.volunteer_batch(volunteers, 0);

        let snapshot = selector.save_snapshot();

        let mut new_selector = CommitteeSelector::new(2);
        assert!(new_selector.load_snapshot(snapshot).is_ok());

        let status = new_selector.check_committee_status(1).unwrap();
        assert_eq!(status.0, CommitteeStatus::AcceptingVolunteers);
        assert_eq!(status.1, vec![123]);
        assert_eq!(status.2, 0);
    }

    #[test]
    fn test_volunteer_double_volunteering() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);
        assert!(selector.create_committee(1, &logger));

        // First volunteer attempt for 101 (should succeed)
        let volunteers_1 = vec![VolunteerKeyDigest {
            public_key_digest: 101,
            key_endorsement_status: EndorsementStatus::Valid as i32,
        }];
        let assignments_1 = selector.volunteer_batch(volunteers_1, 42);
        assert_eq!(assignments_1.len(), 1);
        assert_eq!(assignments_1[0].committee_id, 1);

        // Second volunteer attempt for 101 (should fail/be rejected due to bloom
        // filter)
        let volunteers_2 = vec![VolunteerKeyDigest {
            public_key_digest: 101,
            key_endorsement_status: EndorsementStatus::Valid as i32,
        }];
        let assignments_2 = selector.volunteer_batch(volunteers_2, 42);
        assert_eq!(assignments_2.len(), 1);
        assert_eq!(assignments_2[0].committee_id, -1);
        assert_eq!(assignments_2[0].key_endorsement_status, EndorsementStatus::Duplicate as i32);

        // Check committee status
        let status = selector.check_committee_status(1).unwrap();
        assert_eq!(status.0, CommitteeStatus::AcceptingVolunteers);
        assert_eq!(status.1, vec![101]); // Only added once!
        assert_eq!(status.2, 1); // rejected_volunteers_count is 1
    }

    #[test]
    fn test_rejected_volunteers_count_saturation() {
        let logger = Logger::root(Discard, o!());
        let mut selector = CommitteeSelector::new(2);

        assert!(selector.create_committee(1, &logger));

        // First volunteer for 101 (accepted and added to bloom filter)
        let volunteers_1 = vec![VolunteerKeyDigest {
            public_key_digest: 101,
            key_endorsement_status: EndorsementStatus::Valid as i32,
        }];
        selector.volunteer_batch(volunteers_1, 42);

        // Save snapshot, modify rejected_volunteers_count to usize::MAX, and reload
        let mut snapshot = selector.save_snapshot();
        assert_eq!(snapshot.committees.len(), 1);
        snapshot.committees[0].rejected_volunteers_count = usize::MAX as i64;

        assert!(selector.load_snapshot(snapshot).is_ok());

        // Second volunteer attempt for 101 (duplicate)
        let volunteers_2 = vec![VolunteerKeyDigest {
            public_key_digest: 101,
            key_endorsement_status: EndorsementStatus::Valid as i32,
        }];
        let assignments_2 = selector.volunteer_batch(volunteers_2, 42);
        assert_eq!(assignments_2.len(), 1);
        assert_eq!(assignments_2[0].committee_id, -1);
        assert_eq!(assignments_2[0].key_endorsement_status, EndorsementStatus::Duplicate as i32);

        // Verify rejected_volunteers_count saturated to usize::MAX
        let status = selector.check_committee_status(1).unwrap();
        assert_eq!(status.2, usize::MAX);
    }

    #[test]
    fn test_fingerprint() {
        let data = b"test data";
        let hash1 = fingerprint(data);
        let hash2 = fingerprint(data);
        assert_eq!(hash1, hash2);

        let hash3 = fingerprint(b"different data");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_bloom_filter_try_insert() {
        let mut filter = BloomFilter::new(100);

        // First insert should return true (newly inserted)
        assert!(filter.try_insert(42));

        // Second insert should return false (already present)
        assert!(!filter.try_insert(42));

        // A different value should return true (newly inserted)
        assert!(filter.try_insert(43));

        // 42 should still return false
        assert!(!filter.try_insert(42));
    }
}
