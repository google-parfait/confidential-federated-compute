// Copyright 2025 Google LLC.
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

use std::{
    cmp::Reverse,
    collections::{btree_map, BTreeMap, BinaryHeap},
    ops::Bound::Included,
    sync::atomic::{AtomicI64, Ordering},
};

use anyhow::{anyhow, bail, Context, Result};
use storage_proto::{
    confidential_federated_compute::kms::{
        read_response, update_request, ReadRequest, ReadResponse, UpdateRequest, UpdateResponse,
    },
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::Code;
use tracing::debug;

/// The underlying storage for the KMS: a key-value store with an associated
/// clock.
#[derive(Default)]
pub struct Storage {
    /// The monotonically increasing current time in seconds since the epoch.
    /// This value is only used for entry expiration, so higher resolution is
    /// unnecessary.
    seconds_since_epoch: AtomicI64,

    /// The stored data.
    data: BTreeMap<u128, StorageEntry>,

    /// A collection of entries that have not yet expired, ordered by ascending
    /// expiration time.
    expirations: BinaryHeap<Reverse<ExpirationEntry>>,
}

#[derive(Debug)]
struct StorageEntry {
    value: Vec<u8>,
    expiration_seconds: i64, // 0 if the entry never expires.
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct ExpirationEntry {
    expiration_seconds: i64,
    key: u128,
}

impl Storage {
    /// Returns stored entries.
    pub fn read(&self, request: &ReadRequest) -> Result<ReadResponse> {
        let mut entries = Vec::new();
        for range in &request.ranges {
            let start = Self::parse_key(&range.start)?;
            let end = range.end.as_deref().map(Self::parse_key).transpose()?.unwrap_or(start);
            if end < start {
                return Err(anyhow!("invalid range").context(Code::InvalidArgument));
            }

            for (&key, value) in self.data.range((Included(start), Included(end))) {
                let expiration = match value.expiration_seconds {
                    s if s > 0 => Some(Timestamp { seconds: s, ..Default::default() }),
                    _ => None,
                };
                entries.push(read_response::Entry {
                    key: key.to_be_bytes().to_vec(),
                    value: value.value.clone(),
                    expiration,
                });
            }
        }
        Ok(ReadResponse {
            now: Some(Timestamp {
                seconds: self.seconds_since_epoch.load(Ordering::Relaxed),
                ..Default::default()
            }),
            entries,
        })
    }

    /// Adds, updates, or removes stored entries.
    pub fn update(&mut self, now: &Timestamp, request: UpdateRequest) -> Result<UpdateResponse> {
        // Validate the request before applying any updates.
        for update in &request.updates {
            let entry = self.data.get(&Self::parse_key(&update.key)?);
            if let Some(preconditions) = &update.preconditions {
                Self::check_preconditions(entry, preconditions)
                    .context(format!("preconditions not satisfied for key {:?}", update.key))
                    .context(Code::FailedPrecondition)?;
            }
        }

        // If we've reached this point, all updates can be successfully applied.
        // Update the clock, but don't allow it to go backwards.
        let now_seconds =
            self.seconds_since_epoch.fetch_max(now.seconds, Ordering::Relaxed).max(now.seconds);
        for update in request.updates {
            let key = Self::parse_key(&update.key)?;
            if let Some(value) = update.value {
                let expiration_seconds =
                    update.ttl.as_ref().map(|ttl| now_seconds.saturating_add(ttl.seconds));
                let entry =
                    StorageEntry { value, expiration_seconds: expiration_seconds.unwrap_or(0) };
                debug!("setting entry {:?} = {:?}", key, entry);
                self.data.insert(key, entry);
                if let Some(expiration_seconds) = expiration_seconds {
                    self.expirations.push(Reverse(ExpirationEntry { expiration_seconds, key }));
                }
            } else {
                debug!("removing entry {:?}", key);
                self.data.remove(&key);
            }
        }

        // Remove expired entries.
        while let Some(Reverse(exp)) = self.expirations.peek() {
            if exp.expiration_seconds > now_seconds {
                break;
            }

            // Since existing expiration entries in the heap aren't updated if
            // an expiration time is changed or a StorageEntry is removed, we
            // must also check that the entry exists and is actually expired.
            let exp = self.expirations.pop().unwrap().0;
            if let btree_map::Entry::Occupied(entry) = self.data.entry(exp.key) {
                let expiration_seconds = entry.get().expiration_seconds;
                if expiration_seconds > 0 && expiration_seconds <= now_seconds {
                    debug!("expiring entry {:?}", entry.key());
                    entry.remove_entry();
                }
            }
        }

        Ok(UpdateResponse {})
    }

    /// Parses a key as a big-endian u128.
    fn parse_key(key: &[u8]) -> Result<u128> {
        let key: [u8; 16] = key.try_into().context("invalid key").context(Code::InvalidArgument)?;
        Ok(u128::from_be_bytes(key))
    }

    /// Verifies preconditions for a single update entry.
    fn check_preconditions(
        entry: Option<&StorageEntry>,
        preconditions: &update_request::Preconditions,
    ) -> Result<()> {
        // Check the existence precondition.
        match (preconditions.exists, entry) {
            (Some(true), None) => bail!("exists=true not satisfied"),
            (Some(false), Some(_)) => bail!("exists=false not satisfied"),
            _ => {}
        }

        // Check the value precondition.
        match (&preconditions.value, entry) {
            (Some(value), Some(e)) if *value != e.value => bail!("value not satisfied"),
            (Some(_), None) => bail!("value not satisfied (entry doesn't exist)"),
            _ => {}
        }

        Ok(())
    }
}
