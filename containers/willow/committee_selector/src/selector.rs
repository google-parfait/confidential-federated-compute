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

use crate::apps::willow::committee_selector::service::{
    CommitteeSelectorSnapshot, CommitteeSnapshot, CommitteeStatus, EndorsementStatus,
    VolunteerKeyDigest, VolunteerRecord,
};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use slog::{error, Logger};

/// The minimum number of volunteers a committee must collect before the
/// selection phase can run.
pub const MIN_NUMBER_OF_VOLUNTEERS_PER_COMMITTEE: usize = 1024 * 1024;

/// The final target cohort (or member) size selected for each committee.
pub const NUMBER_OF_MEMBERS_PER_COMMITTEE: usize = 128;

// Bloom filter size: 2 Megabytes. With 4 hash functions and targeting
// MIN_NUMBER_OF_VOLUNTEERS_PER_COMMITTEE (2^20 elements), the false positive
// probability is approximately (1 - e^(-kn/m))^k, where k=4, n=2^20, and m=2^21
// * 8. k * n / m = 4 * 2^20 / (2^21 * 8) = 4 / 16 = 0.25.
// So, p ~= (1 - e^(-0.25))^4 ~= (1 - 0.7788)^4 ~= 0.2212^4 ~= 0.002407 (0.24%).
pub const BLOOM_FILTER_SIZE_BYTES: usize = 2 * 1024 * 1024;

/// A space-efficient, probabilistic set implementation used to track unique
/// volunteers.
///
/// ### Core Rationale:
/// In a TCP / CFC, replicating state expensive. Storing the exact public keys
/// or digests of every volunteer that attempts registration (potentially
/// millions of volunteers across many active committees) is unacceptable as it
/// creates an O(n) space complexity per committee.
///
/// To prevent Sybil or duplicate-joining attacks (where an actor registers the
/// same public key multiple times to bias the random cohort selection), the
/// selector must detect and filter out duplicate registrations. The
/// `BloomFilter` solves this with a constant memory footprint.
#[derive(PartialEq, Debug, Clone)]
pub struct BloomFilter {
    pub data: Vec<u8>,
}

impl BloomFilter {
    pub fn new(size_bytes: usize) -> Self {
        Self { data: alloc::vec![0u8; size_bytes] }
    }

    pub fn try_insert(&mut self, item: i64) -> bool {
        let hashes = self.get_hashes(item);
        let m_bits = (self.data.len() * 8) as u64;
        let mut inserted = false;
        for hash in hashes {
            let bit_idx = (hash % m_bits) as usize;
            let byte_idx = bit_idx / 8;
            let bit_offset = bit_idx % 8;
            let mask = 1 << bit_offset;
            if (self.data[byte_idx] & mask) == 0 {
                self.data[byte_idx] |= mask;
                inserted = true;
            }
        }
        inserted
    }

    fn get_hashes(&self, item: i64) -> [u64; 4] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&item.to_le_bytes());
        let result = hasher.finalize();
        let mut hashes = [0u64; 4];
        for i in 0..4 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&result[i * 8..(i + 1) * 8]);
            hashes[i] = u64::from_le_bytes(bytes);
        }
        hashes
    }
}

/// Represents a single committee (or cohort) during its volunteer collection
/// and member selection lifecycle.
///
/// Each committee accepts a stream of unique volunteer key digests and selects
/// a representative, unbiased, and uniform random sample of members using a
/// deterministic Reservoir Sampling algorithm.
#[derive(PartialEq, Debug)]
pub struct Committee {
    pub members: Vec<i64>,
    pub status: CommitteeStatus,
    pub sequential_order: i64,
    pub total_volunteers_seen: usize,
    pub bloom_filter: BloomFilter,
    pub rejected_volunteers_count: usize,
}

impl Committee {
    /// Safe addition of a volunteer to the committee.
    ///
    /// Evaluates duplicate registration using the Bloom filter, increments the
    /// rejected count if it's a duplicate, or else runs the deterministic
    /// Reservoir Sampling algorithm to decide if the new volunteer is
    /// selected into the `members` reservoir sample.
    ///
    /// Returns `true` if the volunteer was added (i.e. is not a duplicate), and
    /// `false` if the volunteer was rejected as a duplicate.
    pub fn try_add_volunteer(
        &mut self,
        volunteer_public_key_digest: i64,
        randomness: u64,
        committee_id: i64,
    ) -> bool {
        if !self.bloom_filter.try_insert(volunteer_public_key_digest) {
            self.rejected_volunteers_count = self.rejected_volunteers_count.saturating_add(1);
            return false;
        }

        self.total_volunteers_seen += 1;
        let n = self.total_volunteers_seen;
        let k = NUMBER_OF_MEMBERS_PER_COMMITTEE;

        // **Reservoir Sampling Algorithm:**
        // Allows selecting a simple random sample of `k` members from a stream of `n`
        // incoming volunteers of unknown/arbitrary length in a single pass.
        //
        // ### Mathematical Properties:
        // 1. **Uniform Probability:** At any point, every volunteer seen so far has an
        //    equal probability of being in the current sample of size `k`.
        //    - For the first `k` elements (n <= k), we always keep them.
        //    - For any subsequent `n > k` element, we select it to replace an existing
        //      element with probability `k/n`. If selected, it uniformly replaces one
        //      of the `k` items.
        // 2. **Space Efficiency:** Requires only storing `k` elements (128) in the
        //    reservoir, rather than all `n` elements. This results in O(k) memory
        //    complexity instead of O(n), which is crucial for memory-constrained secure
        //    enclaves.
        // 3. **Reproducibility & Verifiability:** The random selection index `j` is
        //    computed using a deterministic hash function over a block of inputs
        //    including a random seed (sent with the batch), the `committee_id`, and a
        //    counter `n`. This makes the random choice reproducible across all nodes in
        //    the distributed nodes.
        if n <= k {
            self.members.push(volunteer_public_key_digest);
        } else {
            let mut reservoir_prng_data = Vec::new();
            reservoir_prng_data.extend_from_slice(&randomness.to_le_bytes());
            reservoir_prng_data.extend_from_slice(&committee_id.to_le_bytes());
            reservoir_prng_data.extend_from_slice(&(n as u64).to_le_bytes());
            let random_val = fingerprint(&reservoir_prng_data);
            let j = (random_val % (n as u64)) as usize;

            if j < k {
                self.members[j] = volunteer_public_key_digest;
            }
        }

        true
    }
}

/// The main state manager responsible for orchestrating the cohort/committee
/// selection workflow.
///
/// ### Core Functions:
/// 1. **Lifecycle Management:** Creates new committees and maintains a bounded
///    map of active/completed committees up to `max_number_of_committees`. When
///    the limit is reached, it performs FIFO eviction using `sequential_order`
///    to free enclave memory.
/// 2. **Random Distribution:** Distributes batches of incoming volunteer
///    registrations across all active committees currently in the
///    `AcceptingVolunteers` state using a pseudo-random assignment.
/// 3. **Cohort Selection:** Processes batch volunteer streams, manages
///    duplicates, and dynamically updates the random reservoir samples for
///    members.
pub struct CommitteeSelector {
    pub(crate) committees: BTreeMap<i64, Committee>,
    pub(crate) max_number_of_committees: usize,
}

impl CommitteeSelector {
    pub fn new(max_number_of_committees: usize) -> Self {
        CommitteeSelector { committees: BTreeMap::new(), max_number_of_committees }
    }

    pub fn create_committee(&mut self, committee_id: i64, logger: &Logger) -> bool {
        if self.committees.contains_key(&committee_id) {
            return false;
        }

        while self.committees.len() >= self.max_number_of_committees && !self.committees.is_empty()
        {
            self.remove_oldest_committee(logger);
        }

        let highest_sequential_order: i64 =
            self.committees.values().map(|committee| committee.sequential_order).max().unwrap_or(0);

        self.committees.insert(
            committee_id,
            Committee {
                status: CommitteeStatus::AcceptingVolunteers,
                sequential_order: highest_sequential_order + 1,
                members: Vec::with_capacity(NUMBER_OF_MEMBERS_PER_COMMITTEE),
                total_volunteers_seen: 0,
                bloom_filter: BloomFilter::new(BLOOM_FILTER_SIZE_BYTES),
                rejected_volunteers_count: 0,
            },
        );
        true
    }

    pub fn volunteer_batch(
        &mut self,
        volunteer_key_digests: Vec<VolunteerKeyDigest>,
        randomness: u64,
    ) -> Vec<VolunteerRecord> {
        volunteer_key_digests
            .into_iter()
            .map(|digest| self.process_volunteer(digest, randomness))
            .collect()
    }

    /// Processes a single volunteer: validates key endorsement status, assigns
    /// the volunteer to an eligible committee, deduplicates via Bloom
    /// filter, and samples members using reservoir sampling.
    fn process_volunteer(
        &mut self,
        volunteer_key_digest: VolunteerKeyDigest,
        randomness: u64,
    ) -> VolunteerRecord {
        let mut volunteer = VolunteerRecord {
            member_public_key_digest: volunteer_key_digest.public_key_digest,
            committee_id: -1,
            key_endorsement_status: volunteer_key_digest.key_endorsement_status,
        };

        if volunteer_key_digest.key_endorsement_status != EndorsementStatus::Valid as i32 {
            return volunteer;
        }

        let mut prng_bytes = [0u8; 16];
        prng_bytes[0..8].copy_from_slice(&randomness.to_le_bytes());
        prng_bytes[8..16].copy_from_slice(&volunteer_key_digest.public_key_digest.to_le_bytes());
        let assignment_randomness = fingerprint(&prng_bytes);

        let Some(committee_id) = self.next_committee_assignment(assignment_randomness) else {
            return volunteer;
        };

        let Some(committee) = self.committees.get_mut(&committee_id) else {
            return volunteer;
        };

        if committee.try_add_volunteer(
            volunteer_key_digest.public_key_digest,
            randomness,
            committee_id,
        ) {
            volunteer.committee_id = committee_id;
        } else {
            volunteer.key_endorsement_status = EndorsementStatus::Duplicate as i32;
        }

        volunteer
    }

    pub fn sample_committee(
        &mut self,
        committee_id: i64,
    ) -> Result<(CommitteeStatus, Vec<i64>, usize), String> {
        let committee = self.committees.get_mut(&committee_id);
        if committee.is_none() {
            return Err(format!("Committee not found for committee id {}", committee_id));
        }

        let committee = committee.unwrap();
        if committee.status != CommitteeStatus::AcceptingVolunteers {
            return Err(format!("Committee {} is not active", committee_id));
        }

        if committee.total_volunteers_seen < MIN_NUMBER_OF_VOLUNTEERS_PER_COMMITTEE {
            return Err(format!("Committee {} does not have enough volunteers", committee_id));
        }

        committee.status = CommitteeStatus::SelectionComplete;

        Ok((committee.status, committee.members.clone(), committee.rejected_volunteers_count))
    }

    pub fn check_committee_status(
        &self,
        committee_id: i64,
    ) -> Option<(CommitteeStatus, Vec<i64>, usize)> {
        self.committees
            .get(&committee_id)
            .map(|c| (c.status, c.members.clone(), c.rejected_volunteers_count))
    }

    /// Finds a random active committee that should accept a volunteer.
    pub fn next_committee_assignment(&mut self, randomness: u64) -> Option<i64> {
        let eligible_committees: Vec<i64> = self
            .committees
            .iter()
            .filter(|(_, committee)| {
                committee.status == CommitteeStatus::AcceptingVolunteers
                    && committee.total_volunteers_seen < usize::MAX
            })
            .map(|(&id, _)| id)
            .collect();

        if eligible_committees.is_empty() {
            return None;
        }

        let index = (randomness % eligible_committees.len() as u64) as usize;
        let selected_id = eligible_committees[index];
        Some(selected_id)
    }

    pub fn remove_oldest_committee(&mut self, logger: &Logger) {
        let oldest = self
            .committees
            .iter()
            .min_by_key(|(_, c)| c.sequential_order)
            .map(|(&id, c)| (id, c.status));

        if let Some((id, status)) = oldest {
            if status == CommitteeStatus::AcceptingVolunteers {
                error!(logger, "Error: removing active committee id {}", id);
            }
            self.committees.remove(&id);
        }
    }

    pub fn save_snapshot(&self) -> CommitteeSelectorSnapshot {
        let mut snapshot =
            CommitteeSelectorSnapshot { committees: Vec::<CommitteeSnapshot>::new() };

        for (committee_id, committee) in &self.committees {
            snapshot.committees.push(CommitteeSnapshot {
                committee_id: committee_id.clone(),
                status: committee.status.clone().into(),
                sequential_order: committee.sequential_order.clone(),
                member_public_key_digests: committee.members.clone(),
                total_volunteers_seen: committee.total_volunteers_seen as i64,
                bloom_filter: committee.bloom_filter.data.clone(),
                rejected_volunteers_count: committee.rejected_volunteers_count as i64,
            });
        }

        snapshot
    }

    pub fn load_snapshot(&mut self, snapshot: CommitteeSelectorSnapshot) -> Result<(), String> {
        self.committees.clear();
        for committee in snapshot.committees {
            let total_volunteers_seen = committee.total_volunteers_seen as usize;
            self.committees.insert(
                committee.committee_id,
                Committee {
                    status: committee.status.try_into().unwrap_or(CommitteeStatus::Unspecified),
                    sequential_order: committee.sequential_order,
                    members: committee.member_public_key_digests,
                    total_volunteers_seen,
                    bloom_filter: BloomFilter { data: committee.bloom_filter },
                    rejected_volunteers_count: committee.rejected_volunteers_count as usize,
                },
            );
        }

        Ok(())
    }
}

/// Helper utility to compute a 64-bit deterministic hash (fingerprint) of a
/// given byte slice.
///
/// Uses SHA-256 internally and reads the first 8 bytes of the finalized hash in
/// little-endian format. Used to derive indexes for Reservoir Sampling and mock
/// deterministic PRNG behavior.
pub fn fingerprint(data: &[u8]) -> u64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&result[..8]);
    u64::from_le_bytes(bytes)
}
