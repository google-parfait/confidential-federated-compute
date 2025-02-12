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

use googletest::prelude::*;
use matchers::{code, has_context};
use storage::Storage;
use storage_proto::{
    confidential_federated_compute::kms::{
        read_request, read_response, update_request, ReadRequest, ReadResponse, UpdateRequest,
    },
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::Code;

/// Creates a ReadRequest that reads all entries from the storage.
fn full_read_request() -> ReadRequest {
    ReadRequest {
        ranges: vec![read_request::Range {
            start: u128::MIN.to_be_bytes().to_vec(),
            end: Some(u128::MAX.to_be_bytes().to_vec()),
        }],
    }
}

#[test_log::test(googletest::test)]
fn default_is_empty() {
    let storage = Storage::default();
    expect_that!(
        storage.read(&full_read_request()),
        ok(eq(ReadResponse { now: Some(Timestamp::default()), ..Default::default() }))
    );
}

#[test_log::test(googletest::test)]
fn read_single_entry() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                },
            ],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range { start: 6u128.to_be_bytes().to_vec(), end: None }],
        }),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            entries: vec![read_response::Entry {
                key: 6u128.to_be_bytes().to_vec(),
                value: b"value 6".into(),
            }],
        }))
    );
}

#[test_log::test(googletest::test)]
fn read_range() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 5u128.to_be_bytes().to_vec(),
                    value: Some(b"value 5".into()),
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                },
                update_request::Update {
                    key: 9u128.to_be_bytes().to_vec(),
                    value: Some(b"value 9".into()),
                },
            ],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range {
                start: 6u128.to_be_bytes().to_vec(),
                end: Some(8u128.to_be_bytes().to_vec()),
            }],
        }),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            entries: vec![
                read_response::Entry {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: b"value 6".into(),
                },
                read_response::Entry {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: b"value 8".into(),
                },
            ],
        }))
    );
}

#[test_log::test(googletest::test)]
fn read_multiple_values() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                },
            ],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![
                read_request::Range { start: 4u128.to_be_bytes().to_vec(), end: None },
                read_request::Range { start: 8u128.to_be_bytes().to_vec(), end: None },
            ],
        }),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            entries: vec![
                read_response::Entry {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: b"value 4".into(),
                },
                read_response::Entry {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: b"value 8".into(),
                },
            ],
        }))
    );
}

#[test_log::test(googletest::test)]
fn empty_read() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 4u128.to_be_bytes().to_vec(),
                value: Some(b"value 4".into()),
            }],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest::default()),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            entries: vec![],
        }))
    );
}

#[test_log::test(googletest::test)]
fn read_invalid_range() {
    let storage = Storage::default();
    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range { start: b"invalid".into(), end: None }],
        }),
        err(all!(code(Code::InvalidArgument), has_context(eq("invalid key"))))
    );
    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range {
                start: 0u128.to_be_bytes().to_vec(),
                end: Some(b"invalid".into()),
            }],
        }),
        err(all!(code(Code::InvalidArgument), has_context(eq("invalid key"))))
    );
    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range {
                start: 5u128.to_be_bytes().to_vec(),
                end: Some(4u128.to_be_bytes().to_vec()),
            }],
        }),
        err(all!(code(Code::InvalidArgument), has_context(eq("invalid range"))))
    );
}

#[test_log::test(googletest::test)]
fn empty_write() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&full_read_request()),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            entries: vec![],
        }))
    );
}

#[test_log::test(googletest::test)]
fn write_invalid_entry() {
    let mut storage = Storage::default();
    expect_that!(
        storage.update(UpdateRequest {
            now: None, // now is required.
            updates: vec![update_request::Update {
                key: 2u128.to_be_bytes().to_vec(),
                value: Some(b"value".into()),
            }],
        }),
        err(all!(code(Code::InvalidArgument), has_context(eq("UpdateRequest missing now"))))
    );
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: b"invalid".into(),
                value: Some(b"value".into()),
            }],
        }),
        err(all!(code(Code::InvalidArgument), has_context(eq("invalid key"))))
    );

    // Since all updates failed, the storage should not be modified.
    expect_that!(
        storage.read(&full_read_request()),
        ok(eq(ReadResponse { now: Some(Timestamp::default()), entries: vec![] }))
    );
}

#[test_log::test(googletest::test)]
fn clock_is_monotonic() {
    let mut storage = Storage::default();
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(storage.clock().get_milliseconds_since_epoch(), eq(200_000));

    // Set a time in the past. The request should succeed, but the clock should not
    // advance backwards.
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(storage.clock().get_milliseconds_since_epoch(), eq(200_000));
}

#[test_log::test(googletest::test)]
fn delete_entry() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                },
            ],
        }),
        ok(anything())
    );

    // Remove an entry by setting its value to None.
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 4u128.to_be_bytes().to_vec(),
                value: None,
            },],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&full_read_request()),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            entries: vec![read_response::Entry {
                key: 6u128.to_be_bytes().to_vec(),
                value: b"value 6".into(),
            },],
        }))
    );
}

#[test_log::test(googletest::test)]
fn delete_missing_entry() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                },
            ],
        }),
        ok(anything())
    );

    // Remove an entry that doesn't exist.
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: None,
            },],
        }),
        ok(anything())
    );

    // The state should be unchanged (except for the current time).
    expect_that!(
        storage.read(&full_read_request()),
        ok(eq(ReadResponse {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            entries: vec![
                read_response::Entry {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: b"value 4".into(),
                },
                read_response::Entry {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: b"value 6".into(),
                },
            ],
        }))
    );
}
