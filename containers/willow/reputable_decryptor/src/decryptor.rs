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

use crate::secure_aggregation::willow::{KeyStateSnapshot, ReputableDecryptorSnapshot};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use prost::bytes::Bytes;
use protobuf::AsView;
use protobuf::{Parse, Serialize};
use status::{self, StatusError};

/// Helper function to convert a C++-backed Google Protobuf message into its
/// Prost counterpart.
pub fn convert_to_prost<PB: Serialize + AsView, PR: prost::Message + Default>(
    pb_msg: &PB,
) -> Result<PR, StatusError> {
    let bytes = pb_msg
        .serialize()
        .map_err(|e| status::internal(&format!("failed to serialize to bytes: {:?}", e)))?;
    PR::decode(&bytes[..])
        .map_err(|e| status::internal(&format!("failed to decode bytes to prost: {:?}", e)))
}

/// Helper function to convert a Prost message into its C++-backed Google
/// Protobuf counterpart.
pub fn convert_from_prost<PR: prost::Message, PB: Parse>(pr_msg: &PR) -> Result<PB, StatusError> {
    let bytes = pr_msg.encode_to_vec();
    PB::parse(&bytes)
        .map_err(|e| status::internal(&format!("failed to parse bytes to protobuf: {:?}", e)))
}

/// Persistent state wrapper combining the compact PRNG seed with a tracking
/// sequential index.
///
/// Rather than storing the full `willow_v1_decryptor::DecryptorState` (which
/// contains the heavy RLWE secret key share `sk_share` consisting of large
/// polynomials), we store only the 32-byte `prng_seed` from which all key
/// materials can be reconstructed on the fly. This keeps the persisted state
/// and snapshot sizes extremely small for the replicated Raft log.
pub struct OrderedDecryptorState {
    pub prng_seed: Vec<u8>,
    pub sequential_order: i64,
}

/// Bounded collection state mapping target keys to their operational states.
pub struct ReputableDecryptorState {
    pub decryptor_states: BTreeMap<String, OrderedDecryptorState>,
    pub max_number_of_decryptor_states: usize,
}

impl Default for ReputableDecryptorState {
    fn default() -> Self {
        Self { decryptor_states: BTreeMap::new(), max_number_of_decryptor_states: 100 }
    }
}

impl ReputableDecryptorState {
    /// Retrieves the mutable state for a key, or constructs it while evicting
    /// the oldest key if limit is reached. Returns a tuple of the mutable
    /// state and the evicted key ID (if any).
    pub fn get_or_insert_key_state(
        &mut self,
        key_id: &str,
        prng_seed: Vec<u8>,
    ) -> (&mut OrderedDecryptorState, Option<String>) {
        let mut evicted = None;
        if !self.decryptor_states.contains_key(key_id) {
            while self.decryptor_states.len() >= self.max_number_of_decryptor_states
                && !self.decryptor_states.is_empty()
            {
                if let Some(k) = self.remove_oldest_key_state() {
                    evicted = Some(k);
                }
            }
            let highest_sequential_order =
                self.decryptor_states.values().map(|k| k.sequential_order).max().unwrap_or(0);
            self.decryptor_states.insert(
                key_id.to_string(),
                OrderedDecryptorState { prng_seed, sequential_order: highest_sequential_order + 1 },
            );
        }
        (self.decryptor_states.get_mut(key_id).unwrap(), evicted)
    }

    fn remove_oldest_key_state(&mut self) -> Option<String> {
        let oldest_key = self
            .decryptor_states
            .iter()
            .min_by_key(|(_, k)| k.sequential_order)
            .map(|(key_id, _)| key_id.clone());
        if let Some(key_id) = oldest_key {
            self.decryptor_states.remove(&key_id);
            Some(key_id)
        } else {
            None
        }
    }

    pub fn save_snapshot(&self) -> Result<ReputableDecryptorSnapshot, StatusError> {
        let mut snapshot = ReputableDecryptorSnapshot::default();
        for (key_id, state) in &self.decryptor_states {
            let mut key_proto = KeyStateSnapshot::default();
            key_proto.key_id = key_id.clone();
            key_proto.sequential_order = state.sequential_order;
            key_proto.decryptor_state = Bytes::from(state.prng_seed.clone());

            snapshot.key_states.push(key_proto);
        }
        Ok(snapshot)
    }

    pub fn load_snapshot(
        &mut self,
        snapshot: ReputableDecryptorSnapshot,
    ) -> Result<(), StatusError> {
        self.decryptor_states.clear();
        for key_proto in snapshot.key_states {
            self.decryptor_states.insert(
                key_proto.key_id,
                OrderedDecryptorState {
                    prng_seed: key_proto.decryptor_state.to_vec(),
                    sequential_order: key_proto.sequential_order,
                },
            );
        }
        Ok(())
    }
}
