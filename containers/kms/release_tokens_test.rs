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

use bssl_crypto::{ec, ecdsa, hpke};
use bssl_utils::{asn1_signature_to_p1363, p1363_signature_to_asn1};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet, ClaimsSetBuilder},
    iana, Algorithm, CborSerializable, CoseEncrypt0, CoseEncrypt0Builder, CoseKey, CoseSign1,
    CoseSign1Builder, Header, HeaderBuilder, KeyType, Label, ProtectedHeader,
};
use googletest::prelude::*;
use key_derivation::{derive_public_keys, HPKE_BASE_X25519_SHA256_AES128GCM, PUBLIC_KEY_CLAIM};
use release_tokens::{
    compute_logical_pipeline_updates, decrypt_release_token, endorse_transform_signing_key,
    verify_release_token, LogicalPipelineUpdate, ENCAPSULATED_KEY_PARAM,
    RELEASE_TOKEN_DST_STATE_PARAM, RELEASE_TOKEN_SRC_STATE_PARAM,
};
use storage_proto::confidential_federated_compute::kms::{
    KeysetKeyValue, PipelineInvocationStateValue,
};

const KEY_PREFIX: &[u8] = b"prefix";

/// Derives the HPKE SenderContext for a node.
fn derive_hpke_sender_context(ikm: &[u8], node_id: u32) -> (hpke::SenderContext, Vec<u8>) {
    let public_keys = derive_public_keys(
        HPKE_BASE_X25519_SHA256_AES128GCM,
        KEY_PREFIX,
        ikm,
        [node_id.to_be_bytes()],
    )
    .expect("derive_public_keys failed");
    let public_key = CoseKey::from_slice(&public_keys[0])
        .expect("CoseKey::from_slice failed")
        .params
        .into_iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
        .and_then(|(_, value)| value.into_bytes().ok())
        .expect("public key missing");
    hpke::SenderContext::new(
        &hpke::Params::new(
            hpke::Kem::X25519HkdfSha256,
            hpke::Kdf::HkdfSha256,
            hpke::Aead::Aes128Gcm,
        ),
        &public_key,
        b"",
    )
    .expect("failed to create SenderContext")
}

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
            .verify(data, &p1363_signature_to_asn1(signature.try_into().unwrap()))),
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
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap().into()
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
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap().into()
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
        .create_signature(b"", |msg| asn1_signature_to_p1363(&other_key.sign(msg)).unwrap().into())
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
            asn1_signature_to_p1363(&transform_signing_key.sign(msg)).unwrap().into()
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

#[googletest::test]
fn decrypt_release_token_succeeds() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );

    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        ok(eq(b"plaintext"))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_with_invalid_key_id() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );
    let mut token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    // Decryption should fail with the wrong prefix.
    token_payload.unprotected.key_id = [b"wrong-prefix", node_id.to_be_bytes().as_slice()].concat();
    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("invalid key id")))
    );

    // Decryption should fail if the key id is too short.
    token_payload.unprotected.key_id = [KEY_PREFIX, &node_id.to_be_bytes()[..3]].concat();
    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("invalid key id")))
    );

    // Decryption should fail if the key id is too long.
    token_payload.unprotected.key_id = [KEY_PREFIX, &node_id.to_be_bytes(), b"x"].concat();
    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("invalid key id")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_with_unauthorized_key_id() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );
    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    // If the key id is not in `dst_node_ids`, decryption should fail.
    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id - 1), Value::from(node_id + 1)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("endorsement doesn't include dst_node_id 1234")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_without_intermediates_key() {
    let invocation_state =
        PipelineInvocationStateValue { intermediates_key: None, ..Default::default() };
    let node_id = 1234;
    let (mut context, encapsulated_key) = derive_hpke_sender_context(&[0; 32], node_id);
    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("PipelineInvocationState missing intermediates_key")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_with_wrong_algorithm() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );

    let token_payload = CoseEncrypt0Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("release token has wrong algorithm")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_without_encapsulated_key() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, _) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );

    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                // Don't attach the encapsulated key.
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("release token missing encapsulated key")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_without_ciphertext() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (_, encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );

    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("release token missing ciphertext")))
    );
}

#[googletest::test]
fn decrypt_release_token_fails_with_decryption_error() {
    let invocation_state = PipelineInvocationStateValue {
        intermediates_key: Some(KeysetKeyValue {
            algorithm: HPKE_BASE_X25519_SHA256_AES128GCM,
            ikm: vec![0; 32],
            ..Default::default()
        }),
        ..Default::default()
    };
    let node_id = 1234;
    let (mut context, mut encapsulated_key) = derive_hpke_sender_context(
        &invocation_state.intermediates_key.as_ref().unwrap().ikm,
        node_id,
    );
    encapsulated_key[0] ^= 0xff;

    let token_payload = CoseEncrypt0Builder::new()
        .protected(Header {
            alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
            ..Default::default()
        })
        .unprotected(
            HeaderBuilder::new()
                .key_id([KEY_PREFIX, &node_id.to_be_bytes()].concat())
                .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                .build(),
        )
        .create_ciphertext(b"plaintext", b"", move |plaintext, aad| context.seal(plaintext, aad))
        .build();

    expect_that!(
        decrypt_release_token(
            &token_payload,
            &[Value::from(node_id)],
            &invocation_state,
            KEY_PREFIX
        ),
        err(displays_as(contains_substring("failed to decrypt release token")))
    );
}

#[googletest::test]
fn compute_logical_pipeline_updates_succeeds() {
    expect_that!(
        compute_logical_pipeline_updates([
            (
                "pipeline1",
                &CoseEncrypt0Builder::new()
                    .protected(
                        HeaderBuilder::new()
                            .value(RELEASE_TOKEN_SRC_STATE_PARAM, Value::Null)
                            .value(RELEASE_TOKEN_DST_STATE_PARAM, Value::Bytes(b"1-A".into()))
                            .build()
                    )
                    .build()
            ),
            (
                "pipeline2",
                &CoseEncrypt0Builder::new()
                    .protected(
                        HeaderBuilder::new()
                            .value(RELEASE_TOKEN_SRC_STATE_PARAM, Value::Bytes(b"2-A".into()))
                            .value(RELEASE_TOKEN_DST_STATE_PARAM, Value::Bytes(b"2-B".into()))
                            .build()
                    )
                    .build()
            ),
        ]),
        ok(unordered_elements_are![
            matches_pattern!(LogicalPipelineUpdate {
                logical_pipeline_name: eq("pipeline1"),
                src_state: none(),
                dst_state: eq(b"1-A"),
            }),
            matches_pattern!(LogicalPipelineUpdate {
                logical_pipeline_name: eq("pipeline2"),
                src_state: some(eq(b"2-A")),
                dst_state: eq(b"2-B"),
            }),
        ])
    );
}

#[googletest::test]
fn compute_logical_pipeline_updates_succeeds_with_empty_list() {
    expect_that!(compute_logical_pipeline_updates([]), ok(empty()));
}

#[googletest::test]
fn compute_logical_pipeline_updates_fails_with_multiple_updates() {
    expect_that!(
        compute_logical_pipeline_updates([
            (
                "pipeline",
                &CoseEncrypt0Builder::new()
                    .protected(
                        HeaderBuilder::new()
                            .value(RELEASE_TOKEN_SRC_STATE_PARAM, Value::Bytes(b"A".into()))
                            .value(RELEASE_TOKEN_DST_STATE_PARAM, Value::Bytes(b"B".into()))
                            .build()
                    )
                    .build()
            ),
            (
                "pipeline",
                &CoseEncrypt0Builder::new()
                    .protected(
                        HeaderBuilder::new()
                            .value(RELEASE_TOKEN_SRC_STATE_PARAM, Value::Bytes(b"B".into()))
                            .value(RELEASE_TOKEN_DST_STATE_PARAM, Value::Bytes(b"C".into()))
                            .build()
                    )
                    .build()
            ),
        ]),
        err(displays_as(contains_substring(
            "multiple release tokens per logical pipeline are not yet supported"
        )))
    );
}

#[googletest::test]
fn compute_logical_pipeline_updates_fails_with_missing_src_state() {
    expect_that!(
        compute_logical_pipeline_updates([(
            "pipeline",
            &CoseEncrypt0Builder::new()
                .protected(
                    HeaderBuilder::new()
                        .value(RELEASE_TOKEN_DST_STATE_PARAM, Value::Bytes(b"dst".into()))
                        .build()
                )
                .build()
        ),]),
        err(displays_as(contains_substring("release token missing src state")))
    );
}

#[googletest::test]
fn compute_logical_pipeline_updates_fails_with_missing_dst_state() {
    expect_that!(
        compute_logical_pipeline_updates([(
            "pipeline",
            &CoseEncrypt0Builder::new()
                .protected(
                    HeaderBuilder::new()
                        .value(RELEASE_TOKEN_SRC_STATE_PARAM, Value::Bytes(b"src".into()))
                        .build()
                )
                .build()
        ),]),
        err(displays_as(contains_substring("release token missing dst state")))
    );
}
