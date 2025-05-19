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

use anyhow::anyhow;
use bssl_crypto::hpke;
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet, ClaimsSetBuilder},
    iana, Algorithm, CborSerializable, CoseKey, CoseSign1, Header, KeyType, Label, ProtectedHeader,
};
use googletest::prelude::*;
use key_derivation::{
    derive_private_keys, derive_public_cwts, derive_public_keys, ACCESS_POLICY_SHA256_CLAIM,
    HPKE_BASE_X25519_SHA256_AES128GCM, PUBLIC_KEY_CLAIM,
};
use oak_proto_rust::oak::crypto::v1::Signature;

struct FakeSigner {}
#[async_trait::async_trait]
impl oak_sdk_containers::Signer for FakeSigner {
    async fn sign(&self, data: &[u8]) -> anyhow::Result<Signature> {
        Ok(Signature { signature: [b"<", data, b">"].concat() })
    }
}

/// Extracts the raw key material from a CoseKey.
fn extract_raw_key(cose_key: &[u8], public: bool) -> Vec<u8> {
    let param = if public { iana::OkpKeyParameter::X } else { iana::OkpKeyParameter::D };
    CoseKey::from_slice(cose_key)
        .expect("failed to parse CoseKey")
        .params
        .into_iter()
        .find(|(label, _)| label == &Label::Int(param as i64))
        .map(|(_, value)| value.into_bytes().unwrap())
        .expect("failed to extract key")
}

#[googletest::test]
fn derive_private_keys_produces_cose_key() {
    let keys =
        derive_private_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"info-hash"]);
    assert_that!(keys, ok(elements_are!(not(empty()))));
    expect_that!(
        CoseKey::from_slice(&keys.unwrap()[0]),
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::OKP)),
            alg: some(eq(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))),
            key_id: eq(b"key-idinfo"),
            params: unordered_elements_are![
                (
                    eq(Label::Int(iana::OkpKeyParameter::Crv as i64)),
                    eq(Value::from(iana::EllipticCurve::X25519 as u64)),
                ),
                (
                    eq(Label::Int(iana::OkpKeyParameter::D as i64)),
                    matches_pattern!(Value::Bytes(not(empty()))),
                ),
            ],
        })),
    );
}

#[googletest::test]
fn derive_private_keys_varies_by_input() {
    let extract_private_key = |cose_key| extract_raw_key(cose_key, /* public= */ false);

    let keys = derive_private_keys(
        HPKE_BASE_X25519_SHA256_AES128GCM,
        b"id",
        &[0; 32],
        [b"foo", b"bar", b"foo"],
    );
    assert_that!(keys, ok(elements_are!(not(empty()), not(empty()), not(empty()))));
    let keys = keys.unwrap();
    // Different info values should result in different keys.
    expect_that!(extract_private_key(&keys[0]), not(eq(extract_private_key(&keys[1]))));
    expect_that!(extract_private_key(&keys[0]), eq(extract_private_key(&keys[2])));

    // Different ikm should result in different keys.
    let keys2 = derive_private_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"id", &[1; 32], [b"foo"]);
    assert_that!(keys2, ok(len(eq(1))));
    expect_that!(
        extract_private_key(&keys2.as_ref().unwrap()[0]),
        not(eq(extract_private_key(&keys[0]))),
    );

    // The key id should not affect the key material.
    let keys3 =
        derive_private_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"other-id", &[0; 32], [b"foo"]);
    assert_that!(keys3, ok(len(eq(1))));
    expect_that!(
        extract_private_key(&keys3.as_ref().unwrap()[0]),
        eq(extract_private_key(&keys[0])),
    );
}

#[googletest::test]
fn derive_private_keys_is_deterministic() {
    let keys =
        derive_private_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"info-hash"]);
    assert_that!(keys, ok(elements_are!(not(empty()))));
    expect_that!(
        extract_raw_key(&keys.unwrap()[0], /* public= */ false),
        eq(&[
            255, 73, 220, 113, 69, 182, 224, 57, 232, 237, 92, 83, 88, 115, 165, 40, 76, 194, 197,
            114, 237, 73, 140, 62, 130, 52, 130, 252, 249, 17, 241, 138
        ])
    );
}

#[googletest::test]
fn derive_private_keys_fails_with_invalid_algorithm() {
    expect_that!(
        derive_private_keys(0, b"key-id", &[0; 32], [b"foo"]),
        err(displays_as(contains_substring("unsupported algorithm"))),
    );
}

#[googletest::test]
fn derive_public_keys_produces_cose_key() {
    let public_keys =
        derive_public_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"foo"]);
    assert_that!(public_keys, ok(elements_are!(not(empty()))));

    expect_that!(
        CoseKey::from_slice(&public_keys.unwrap()[0]),
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::OKP)),
            alg: some(eq(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))),
            key_id: eq(b"key-idfoo"),
            params: unordered_elements_are![
                (
                    eq(Label::Int(iana::OkpKeyParameter::Crv as i64)),
                    eq(Value::from(iana::EllipticCurve::X25519 as u64)),
                ),
                (
                    eq(Label::Int(iana::OkpKeyParameter::X as i64)),
                    matches_pattern!(Value::Bytes(not(empty()))),
                ),
            ],
        })),
    );
}

#[googletest::test]
fn derive_public_keys_fails_with_invalid_algorithm() {
    expect_that!(
        derive_public_keys(0, b"key-id", &[0; 32], [b"foo"]),
        err(displays_as(contains_substring("unsupported algorithm"))),
    );
}

#[googletest::test]
#[tokio::test]
async fn derive_public_cwts_produces_cwt() {
    let cwts = derive_public_cwts(
        HPKE_BASE_X25519_SHA256_AES128GCM,
        b"key-id",
        &[0; 32],
        ClaimsSetBuilder::new().issuer("test".into()).build(),
        &[b"foo"],
        &FakeSigner {},
    )
    .await;
    assert_that!(cwts, ok(elements_are!(not(empty()))));

    let cwt = CoseSign1::from_slice(&cwts.unwrap()[0]);
    assert_that!(
        cwt,
        ok(matches_pattern!(CoseSign1 {
            protected: matches_pattern!(ProtectedHeader {
                header: matches_pattern!(Header {
                    alg: some(eq(Algorithm::Assigned(iana::Algorithm::ES256))),
                }),
            }),
            payload: some(anything()),
        }))
    );
    expect_that!(
        cwt.as_ref().unwrap().verify_signature(b"", |signature, data| {
            if signature == [b"<", data, b">"].concat() {
                Ok(())
            } else {
                Err(anyhow!("invalid signature"))
            }
        }),
        ok(anything()),
    );

    let claims = ClaimsSet::from_slice(&cwt.unwrap().payload.unwrap());
    let public_keys =
        derive_public_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"foo"]);
    assert_that!(
        claims,
        ok(matches_pattern!(ClaimsSet {
            issuer: some(eq("test")),
            rest: all![
                contains((
                    eq(ClaimName::PrivateUse(PUBLIC_KEY_CLAIM)),
                    matches_pattern!(Value::Bytes(eq(public_keys.unwrap()[0].as_slice()))),
                )),
                contains((
                    eq(ClaimName::PrivateUse(ACCESS_POLICY_SHA256_CLAIM)),
                    matches_pattern!(Value::Bytes(eq(b"foo"))),
                )),
            ],
        })),
    );
}

#[googletest::test]
#[tokio::test]
async fn derive_public_cwts_fails_with_invalid_algorithm() {
    expect_that!(
        derive_public_cwts(
            0,
            b"key-id",
            &[0; 32],
            ClaimsSet::default(),
            &[b"foo"],
            &FakeSigner {},
        )
        .await,
        err(displays_as(contains_substring("unsupported algorithm"))),
    );
}

#[googletest::test]
fn public_and_private_keys_are_paired() {
    let private_keys =
        derive_private_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"foo"]);
    assert_that!(private_keys, ok(elements_are!(anything())));
    let recipient_private_key =
        extract_raw_key(&private_keys.unwrap()[0], /* public= */ false);

    let public_keys =
        derive_public_keys(HPKE_BASE_X25519_SHA256_AES128GCM, b"key-id", &[0; 32], [b"foo"]);
    assert_that!(public_keys, ok(elements_are!(anything())));
    let recipient_public_key = extract_raw_key(&public_keys.unwrap()[0], /* public= */ true);

    let params = hpke::Params::new(
        hpke::Kem::X25519HkdfSha256,
        hpke::Kdf::HkdfSha256,
        hpke::Aead::Aes128Gcm,
    );
    let (mut sender_context, encapsulated_key) =
        hpke::SenderContext::new(&params, &recipient_public_key, b"info").unwrap();
    expect_that!(
        hpke::RecipientContext::new(&params, &recipient_private_key, &encapsulated_key, b"info")
            .unwrap()
            .open(&sender_context.seal(b"plaintext", b"aad"), b"aad"),
        some(eq(b"plaintext")),
    );
}
