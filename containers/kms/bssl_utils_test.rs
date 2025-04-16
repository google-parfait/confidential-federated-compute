// Copyright 2025 The Trusted Computations Platform Authors.
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

use bssl_utils::{asn1_signature_to_p1363, p1363_signature_to_asn1};
use googletest::prelude::*;

// While signatures with r or s coordinates that start with \0 are rare, we
// intentionaly test with one to make sure that the conversions correctly handle
// the omitted \0 byte in the DER encoding.

#[googletest::test]
fn asn1_signature_to_p1363_success() {
    let signature = [
        &[48, 67, 2, 31],
        Vec::from_iter(1..32).as_slice(),
        &[2, 32],
        Vec::from_iter(32..64).as_slice(),
    ]
    .concat();
    let expected = Vec::from_iter(0..64);
    expect_that!(asn1_signature_to_p1363(&signature), some(eq(expected)));
}

#[googletest::test]
fn asn1_signature_to_p1363_fails_with_invalid_signature() {
    expect_that!(asn1_signature_to_p1363(b"invalid"), none());
}

#[googletest::test]
fn p1363_signature_to_asn1_success() {
    let signature = Vec::from_iter(0..64);
    let expected = [
        &[48, 67, 2, 31],
        Vec::from_iter(1..32).as_slice(),
        &[2, 32],
        Vec::from_iter(32..64).as_slice(),
    ]
    .concat();
    expect_that!(p1363_signature_to_asn1(&signature), some(eq(expected)));
}

#[googletest::test]
fn p1363_signature_to_asn1_fails_with_invalid_signature() {
    expect_that!(p1363_signature_to_asn1(b"invalid"), none());
}
