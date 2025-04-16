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

use bssl_crypto::{ec, ecdsa};
use bssl_utils::{asn1_signature_to_p1363, p1363_signature_to_asn1};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet, ClaimsSetBuilder},
    iana, Algorithm, CborSerializable, CoseEncrypt0, CoseEncrypt0Builder, CoseKey, CoseSign1,
    CoseSign1Builder, Header, HeaderBuilder, KeyType, Label, ProtectedHeader,
};
use googletest::prelude::*;
use key_derivation::PUBLIC_KEY_CLAIM;
use release_tokens::{endorse_transform_signing_key, verify_release_token};

#[googletest::test]
fn endorse_transform_signing_key_succeeds() {
    let cluster_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let endorsement = endorse_transform_signing_key(
        transform_signing_key.to_public_key().to_x962_uncompressed().as_ref(),
        &cluster_key,
        ClaimsSetBuilder::new().issuer("test".into()).build(),
    )
    .expect("endorse_transform_signing_key failed");

    let cwt = CoseSign1::from_slice(&endorsement);
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
        cwt.as_ref().unwrap().verify_signature(b"", |signature, data| cluster_key
            .to_public_key()
            .verify(data, &p1363_signature_to_asn1(signature).expect("invalid signature"))),
        ok(anything())
    );

    let claims = ClaimsSet::from_slice(&cwt.unwrap().payload.unwrap());
    assert_that!(
        claims,
        ok(matches_pattern!(ClaimsSet {
            issuer: some(eq("test")),
            rest: contains((
                eq(ClaimName::PrivateUse(PUBLIC_KEY_CLAIM)),
                matches_pattern!(Value::Bytes(anything()))
            )),
        }))
    );
    let cose_key = claims
        .unwrap()
        .rest
        .iter()
        .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
        .map(|(_, value)| CoseKey::from_slice(value.as_bytes().unwrap()))
        .unwrap();
    assert_that!(
        cose_key,
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::EC2)),
            alg: some(eq(Algorithm::Assigned(iana::Algorithm::ES256))),
            params: unordered_elements_are![
                (
                    eq(Label::Int(iana::Ec2KeyParameter::Crv as i64)),
                    eq(Value::from(iana::EllipticCurve::P_256 as u64)),
                ),
                (
                    eq(Label::Int(iana::Ec2KeyParameter::X as i64)),
                    matches_pattern!(Value::Bytes(len(eq(32)))),
                ),
                (
                    eq(Label::Int(iana::Ec2KeyParameter::Y as i64)),
                    matches_pattern!(Value::Bytes(len(eq(32)))),
                ),
            ],
        }))
    );
    let cose_key_params: std::collections::BTreeMap<_, _> =
        cose_key.unwrap().params.into_iter().collect();
    // X9.62 has the format "\x04<x><y>".
    let x962_public_key = [
        b"\x04",
        cose_key_params
            .get(&Label::Int(iana::Ec2KeyParameter::X as i64))
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_slice(),
        cose_key_params
            .get(&Label::Int(iana::Ec2KeyParameter::Y as i64))
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_slice(),
    ]
    .concat();
    expect_that!(
        x962_public_key,
        eq(transform_signing_key.to_public_key().to_x962_uncompressed().as_ref()),
    );
}

#[googletest::test]
fn verify_release_token_succeeds() {
    let cluster_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let payload =
        CoseEncrypt0Builder::new().ciphertext(b"ciphertext".into()).build().to_vec().unwrap();
    let token = CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(payload.clone())
        .create_signature(b"", |msg| {
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap()
        })
        .build()
        .to_vec()
        .unwrap();
    let endorsement = endorse_transform_signing_key(
        transform_signing_key.to_public_key().to_x962_uncompressed().as_ref(),
        &cluster_key,
        ClaimsSetBuilder::new().issuer("test".into()).build(),
    )
    .expect("endorse_transform_signing_key failed");

    let (token_payload, claims, verify_signature_fn) =
        verify_release_token(&token, &endorsement).expect("verify_release_token failed");
    expect_that!(token_payload, eq(CoseEncrypt0::from_slice(&payload).unwrap()));
    expect_that!(claims, matches_pattern!(ClaimsSet { issuer: some(eq("test")) }));
    // The signature verification function should succeed with the right key
    // and fail with the wrong one.
    expect_that!(verify_signature_fn(&cluster_key.to_public_key()), ok(anything()));
    expect_that!(verify_signature_fn(&transform_signing_key.to_public_key()), err(anything()));
}

#[googletest::test]
fn verify_release_token_fails_with_invalid_endorsement() {
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let token = CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(CoseEncrypt0::default().to_vec().unwrap())
        .create_signature(b"", |msg| {
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap()
        })
        .build()
        .to_vec()
        .unwrap();

    expect_that!(
        verify_release_token(&token, b"invalid").err(),
        some(displays_as(contains_substring("failed to parse endorsement")))
    );
}

#[googletest::test]
fn verify_release_token_fails_with_invalid_token() {
    let cluster_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let endorsement = endorse_transform_signing_key(
        transform_signing_key.to_public_key().to_x962_uncompressed().as_ref(),
        &cluster_key,
        ClaimsSet::default(),
    )
    .expect("endorse_transform_signing_key failed");

    expect_that!(
        verify_release_token(b"invalid", &endorsement).err(),
        some(displays_as(contains_substring("failed to parse release token")))
    );
}

#[googletest::test]
fn verify_release_token_fails_with_invalid_token_signature() {
    let cluster_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let other_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let token = CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(CoseEncrypt0::default().to_vec().unwrap())
        .create_signature(b"", |msg| asn1_signature_to_p1363(&other_key.sign(msg)).unwrap())
        .build()
        .to_vec()
        .unwrap();
    let endorsement = endorse_transform_signing_key(
        transform_signing_key.to_public_key().to_x962_uncompressed().as_ref(),
        &cluster_key,
        ClaimsSet::default(),
    )
    .expect("endorse_transform_signing_key failed");

    expect_that!(
        verify_release_token(&token, &endorsement).err(),
        some(displays_as(contains_substring("invalid release token signature")))
    );
}

#[googletest::test]
fn verify_release_token_fails_with_invalid_token_payload() {
    let cluster_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let transform_signing_key = ecdsa::PrivateKey::<ec::P256>::generate();
    let token = CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(b"invalid".into())
        .create_signature(b"", |msg| {
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap()
        })
        .build()
        .to_vec()
        .unwrap();
    let endorsement = endorse_transform_signing_key(
        transform_signing_key.to_public_key().to_x962_uncompressed().as_ref(),
        &cluster_key,
        ClaimsSetBuilder::new().issuer("test".into()).build(),
    )
    .expect("endorse_transform_signing_key failed");

    expect_that!(
        verify_release_token(&token, &endorsement).err(),
        some(displays_as(contains_substring("invalid release token payload")))
    );
}
