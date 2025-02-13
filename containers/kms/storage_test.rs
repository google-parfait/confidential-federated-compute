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
    duration_proto::google::protobuf::Duration,
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
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                    ..Default::default()
                },
            ],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![read_request::Range { start: 6u128.to_be_bytes().to_vec(), end: None }],
        }),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(6u128.to_be_bytes()),
                value: eq(b"value 6"),
            })],
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
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 9u128.to_be_bytes().to_vec(),
                    value: Some(b"value 9".into()),
                    ..Default::default()
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
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![
                matches_pattern!(read_response::Entry {
                    key: eq(6u128.to_be_bytes()),
                    value: eq(b"value 6"),
                }),
                matches_pattern!(read_response::Entry {
                    key: eq(8u128.to_be_bytes()),
                    value: eq(b"value 8"),
                }),
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
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                    ..Default::default()
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
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![
                matches_pattern!(read_response::Entry {
                    key: eq(4u128.to_be_bytes()),
                    value: eq(b"value 4"),
                }),
                matches_pattern!(read_response::Entry {
                    key: eq(8u128.to_be_bytes()),
                    value: eq(b"value 8"),
                }),
            ],
        }))
    );
}

#[test_log::test(googletest::test)]
fn read_entry_multiple_times() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                    ..Default::default()
                },
            ],
        }),
        ok(anything())
    );

    // If the same entry matches multiple ranges, it may be returned multiple times.
    expect_that!(
        storage.read(&ReadRequest {
            ranges: vec![
                read_request::Range {
                    start: 4u128.to_be_bytes().to_vec(),
                    end: Some(6u128.to_be_bytes().to_vec())
                },
                read_request::Range { start: 6u128.to_be_bytes().to_vec(), end: None },
                read_request::Range {
                    start: 6u128.to_be_bytes().to_vec(),
                    end: Some(8u128.to_be_bytes().to_vec())
                },
            ],
        }),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: all!(
                contains(matches_pattern!(read_response::Entry {
                    key: eq(4u128.to_be_bytes()),
                    value: eq(b"value 4"),
                }))
                .times(eq(1)),
                contains(matches_pattern!(read_response::Entry {
                    key: eq(6u128.to_be_bytes()),
                    value: eq(b"value 6"),
                }))
                .times(any!(eq(1), eq(3))),
                contains(matches_pattern!(read_response::Entry {
                    key: eq(8u128.to_be_bytes()),
                    value: eq(b"value 8"),
                }))
                .times(eq(1)),
                // The response should not contain any other entries.
                contains(matches_pattern!(read_response::Entry {
                    key: not(any!(
                        eq(4u128.to_be_bytes()),
                        eq(6u128.to_be_bytes()),
                        eq(8u128.to_be_bytes())
                    )),
                }))
                .times(eq(0)),
            ),
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
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&ReadRequest::default()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![],
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
fn multiple_writes_to_same_key() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4a".into()),
                    ..Default::default()
                },
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4b".into()),
                    ..Default::default()
                },
            ],
        }),
        ok(anything())
    );

    // One of the two writes should be applied, but it's unspecified which.
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(4u128.to_be_bytes()),
                value: any!(eq(b"value 4a"), eq(b"value 4b")),
            }),],
        }))
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
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![],
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
                ..Default::default()
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
                ..Default::default()
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
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
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
                ..Default::default()
            },],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(6u128.to_be_bytes()),
                value: eq(b"value 6"),
            })],
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
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ..Default::default()
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
                ..Default::default()
            },],
        }),
        ok(anything())
    );

    // The state should be unchanged (except for the current time).
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![
                matches_pattern!(read_response::Entry {
                    key: eq(4u128.to_be_bytes()),
                    value: eq(b"value 4"),
                }),
                matches_pattern!(read_response::Entry {
                    key: eq(6u128.to_be_bytes()),
                    value: eq(b"value 6"),
                }),
            ],
        }))
    );
}

#[test_log::test(googletest::test)]
fn expired_entry_is_removed() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 4u128.to_be_bytes().to_vec(),
                    value: Some(b"value 4".into()),
                    ttl: None, // never expires
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    ttl: Some(Duration { seconds: 10, ..Default::default() }),
                    ..Default::default()
                },
                update_request::Update {
                    key: 8u128.to_be_bytes().to_vec(),
                    value: Some(b"value 8".into()),
                    ttl: Some(Duration { seconds: 20, ..Default::default() }),
                    ..Default::default()
                },
            ],
        }),
        ok(anything())
    );

    // Advance the clock so that one entry expires.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 110, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(110) })),
            entries: elements_are![
                matches_pattern!(read_response::Entry {
                    key: eq(4u128.to_be_bytes()),
                    value: eq(b"value 4"),
                    expiration: none(),
                }),
                matches_pattern!(read_response::Entry {
                    key: eq(8u128.to_be_bytes()),
                    value: eq(b"value 8"),
                    expiration: some(matches_pattern!(Timestamp { seconds: eq(120) })),
                }),
            ],
        }))
    );
}

#[test_log::test(googletest::test)]
fn expiration_uses_latest_time() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );

    // Insert an entry with a `now` value in the past. The entry should use the
    // Storage's maximum time (200), not the request's time (100).
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value".into()),
                ttl: Some(Duration { seconds: 10, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value"),
                expiration: some(matches_pattern!(Timestamp { seconds: eq(210) })),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn ttl_can_be_shortened() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ttl: Some(Duration { seconds: 100, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Shorten the entry's TTL (200 -> 160).
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 110, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                ttl: Some(Duration { seconds: 50, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(110) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
                expiration: some(matches_pattern!(Timestamp { seconds: eq(160) })),
            })],
        }))
    );

    // Advance the clock so that the entry expires.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 170, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(170) })),
            entries: elements_are![],
        }))
    );

    // Advance the clock to the original expiration time. Nothing should break.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![],
        }))
    );
}

#[test_log::test(googletest::test)]
fn ttl_can_be_extended() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ttl: Some(Duration { seconds: 20, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Extend the entry's TTL (120 -> 210).
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 110, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                ttl: Some(Duration { seconds: 100, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(110) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
                expiration: some(matches_pattern!(Timestamp { seconds: eq(210) })),
            })],
        }))
    );

    // Advance the clock to the original expiration time. The entry should not be
    // removed.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 120, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(120) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
                expiration: some(matches_pattern!(Timestamp { seconds: eq(210) })),
            })],
        }))
    );

    // Advance the clock to the updated expiration time. The entry should be
    // removed.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 210, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(210) })),
            entries: elements_are![],
        }))
    );
}

#[test_log::test(googletest::test)]
fn ttl_can_be_removed() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ttl: Some(Duration { seconds: 20, ..Default::default() }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Remove the entry's TTL.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 110, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                ttl: None,
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(110) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
                expiration: none(),
            })],
        }))
    );

    // Advance the clock to the original expiration time. The entry should not be
    // removed.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 120, ..Default::default() }),
            updates: vec![],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(120) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
                expiration: none(),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn exists_precondition_is_met() {
    let mut storage = Storage::default();

    // Apply an update with `exists=false`.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                preconditions: Some(update_request::Preconditions {
                    exists: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value A"),
            })],
        }))
    );

    // Apply an update with `exists=true`.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    exists: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn exists_precondition_is_not_met() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Attempt an update with `exists=false`.
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    exists: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        err(all!(code(Code::FailedPrecondition), has_context(eq("exists=false not satisfied"))))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value A"),
            })],
        }))
    );

    // Attempt an update with `exists=true`.
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 300, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 6u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    exists: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        err(all!(code(Code::FailedPrecondition), has_context(eq("exists=true not satisfied"))))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value A"),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn value_precondition_is_met() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Apply an update with a satisfied `value` condition.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    value: Some(b"value A".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        ok(anything())
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(200) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value B"),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn value_precondition_is_not_met() {
    let mut storage = Storage::default();
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value A".into()),
                ..Default::default()
            }],
        }),
        ok(anything())
    );

    // Attempt an update with an unsatisfied `value` condition.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    value: Some(b"wrong value".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        err(all!(code(Code::FailedPrecondition), has_context(eq("value not satisfied"))))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value A"),
            })],
        }))
    );

    // Attempt an update with an unsatisfied `value` condition because the entry
    // doesn't exist.
    assert_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 200, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 6u128.to_be_bytes().to_vec(),
                value: Some(b"value B".into()),
                preconditions: Some(update_request::Preconditions {
                    value: Some(b"".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }),
        err(all!(
            code(Code::FailedPrecondition),
            has_context(eq("value not satisfied (entry doesn't exist)"))
        ))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(100) })),
            entries: elements_are![matches_pattern!(read_response::Entry {
                key: eq(5u128.to_be_bytes()),
                value: eq(b"value A"),
            })],
        }))
    );
}

#[test_log::test(googletest::test)]
fn all_preconditions_for_entry_must_be_met() {
    let mut storage = Storage::default();
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![update_request::Update {
                key: 5u128.to_be_bytes().to_vec(),
                value: Some(b"value".into()),
                preconditions: Some(update_request::Preconditions {
                    exists: Some(false),
                    value: Some(b"wrong value".into()),
                }),
                ..Default::default()
            }],
        }),
        err(all!(code(Code::FailedPrecondition)))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(0) })),
            entries: elements_are![],
        }))
    );
}

#[test_log::test(googletest::test)]
fn preconditions_for_all_entries_must_be_met() {
    let mut storage = Storage::default();
    expect_that!(
        storage.update(UpdateRequest {
            now: Some(Timestamp { seconds: 100, ..Default::default() }),
            updates: vec![
                update_request::Update {
                    key: 5u128.to_be_bytes().to_vec(),
                    value: Some(b"value 5".into()),
                    preconditions: Some(update_request::Preconditions {
                        exists: Some(false),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                update_request::Update {
                    key: 6u128.to_be_bytes().to_vec(),
                    value: Some(b"value 6".into()),
                    preconditions: Some(update_request::Preconditions {
                        exists: Some(true),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        }),
        err(all!(code(Code::FailedPrecondition)))
    );
    expect_that!(
        storage.read(&full_read_request()),
        ok(matches_pattern!(ReadResponse {
            now: some(matches_pattern!(Timestamp { seconds: eq(0) })),
            entries: elements_are![],
        }))
    );
}
