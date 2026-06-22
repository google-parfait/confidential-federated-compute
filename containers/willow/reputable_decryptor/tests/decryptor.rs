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

extern crate alloc;
extern crate willow_reputable_decryptor_service;

mod test {
    use alloc::vec;
    use willow_reputable_decryptor_service::decryptor::ReputableDecryptorState;

    #[test]
    fn test_state_snapshot_roundtrip() {
        let mut state = ReputableDecryptorState::default();
        let seed1 = vec![1; 32];
        let seed2 = vec![2; 32];

        // Setup Key 1 and Key 2
        let _ = state.get_or_insert_key_state("key1", seed1.clone());
        let _ = state.get_or_insert_key_state("key2", seed2.clone());

        // Save state snapshot
        let snapshot = state.save_snapshot();
        assert!(snapshot.is_ok());
        let snapshot = snapshot.unwrap();

        // Create a new fresh state and load state snapshot
        let mut state2 = ReputableDecryptorState::default();
        let load_status = state2.load_snapshot(snapshot);
        assert!(load_status.is_ok());

        // Verify state elements successfully loaded and match
        assert!(state2.decryptor_states.contains_key("key1"));
        assert!(state2.decryptor_states.contains_key("key2"));

        let key_state1_orig = state.decryptor_states.get("key1").unwrap();
        let key_state1_restored = state2.decryptor_states.get("key1").unwrap();

        assert_eq!(key_state1_orig.prng_seed, key_state1_restored.prng_seed);
        assert_eq!(key_state1_restored.prng_seed, seed1);
    }

    #[test]
    fn test_key_limit_and_fifo_eviction() {
        let mut state = ReputableDecryptorState::default();
        state.max_number_of_decryptor_states = 2;
        let seed1 = vec![1; 32];
        let seed2 = vec![2; 32];
        let seed3 = vec![3; 32];

        // Setup DKG entries for key1 and key2
        let _ = state.get_or_insert_key_state("key1", seed1);
        let _ = state.get_or_insert_key_state("key2", seed2);

        assert!(state.decryptor_states.contains_key("key1"));
        assert!(state.decryptor_states.contains_key("key2"));

        // Setup key3 (causes FIFO key cleanup of the oldest key "key1"!)
        let _ = state.get_or_insert_key_state("key3", seed3);

        assert!(!state.decryptor_states.contains_key("key1"), "key1 should be evicted");
        assert!(state.decryptor_states.contains_key("key2"));
        assert!(state.decryptor_states.contains_key("key3"));
    }
}
