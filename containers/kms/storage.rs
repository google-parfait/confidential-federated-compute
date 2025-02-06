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
    collections::BTreeMap,
    ops::Bound::Included,
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
};

use anyhow::{anyhow, Context, Result};
use log::debug;
use storage_proto::{
    confidential_federated_compute::kms::{
        read_response, ReadRequest, ReadResponse, UpdateRequest, UpdateResponse,
    },
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::Code;

/// The underlying storage for the KMS: a key-value store with an associated
/// clock.
// TODO: b/393146003 - Add support for update preconditions and entry expiration.
#[derive(Default)]
pub struct Storage {
    /// The monotonically increasing current time in seconds since the epoch.
    clock: Arc<Clock>,

    /// The stored data.
    data: BTreeMap<u128, StorageEntry>,
}

#[derive(Debug)]
struct StorageEntry {
    value: Vec<u8>,
}

impl Storage {
    pub fn clock(&self) -> Arc<dyn oak_attestation_verification_types::util::Clock> {
        self.clock.clone()
    }

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
                entries.push(read_response::Entry {
                    key: key.to_be_bytes().to_vec(),
                    value: value.value.clone(),
                });
            }
        }
        Ok(ReadResponse {
            now: Some(Timestamp {
                seconds: self.clock.seconds_since_epoch(),
                ..Default::default()
            }),
            entries,
        })
    }

    /// Adds, updates, or removes stored entries.
    pub fn update(&mut self, request: UpdateRequest) -> Result<UpdateResponse> {
        // Validate the request before applying any updates.
        let now_seconds = request
            .now
            .map(|t| t.seconds)
            .ok_or_else(|| anyhow!("UpdateRequest missing now").context(Code::InvalidArgument))?;
        for update in &request.updates {
            Self::parse_key(&update.key)?;
        }

        // If we've reached this point, all updates can be successfully applied.
        self.clock.update(now_seconds);
        for update in request.updates {
            let key = Self::parse_key(&update.key)?;
            if let Some(value) = update.value {
                let entry = StorageEntry { value };
                debug!("setting entry {:?} = {:?}", key, entry);
                self.data.insert(key, entry);
            } else {
                debug!("removing entry {:?}", key);
                self.data.remove(&key);
            }
        }

        Ok(UpdateResponse {})
    }

    /// Parses a key as a big-endian u128.
    fn parse_key(key: &[u8]) -> Result<u128> {
        let key: [u8; 16] = key.try_into().context("invalid key").context(Code::InvalidArgument)?;
        Ok(u128::from_be_bytes(key))
    }
}

#[derive(Default)]
struct Clock {
    /// The monotonically increasing current time in seconds since the epoch.
    /// This value is used for entry expiration and verifying the time
    /// ranges in Oak endorsements, so higher resolution is unnecessary.
    seconds_since_epoch: AtomicI64,
}

impl Clock {
    fn seconds_since_epoch(&self) -> i64 {
        self.seconds_since_epoch.load(Ordering::Relaxed)
    }

    fn update(&self, seconds_since_epoch: i64) -> i64 {
        // Don't allow the clock to go backwards.
        self.seconds_since_epoch
            .fetch_max(seconds_since_epoch, Ordering::Relaxed)
            .max(seconds_since_epoch)
    }
}

impl oak_attestation_verification_types::util::Clock for Clock {
    fn get_milliseconds_since_epoch(&self) -> i64 {
        self.seconds_since_epoch().saturating_mul(1000)
    }
}
