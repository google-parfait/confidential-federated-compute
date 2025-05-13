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

use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc,
};

use access_policy_proto::{
    any_proto::google::protobuf::Any,
    fcp::confidentialcompute::{
        pipeline_variant_policy::Transform, ApplicationMatcher,
        DataAccessPolicy as AuthorizedLogicalPipelinePolicies, LogicalPipelinePolicy,
        PipelineVariantPolicy,
    },
    reference_value_proto::oak::attestation::v1::ReferenceValues,
};
use anyhow::Context;
use bssl_crypto::{digest::Sha256, ec, ecdsa, hpke};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet, Timestamp as CwtTimestamp},
    iana, Algorithm, CborSerializable, CoseEncrypt0Builder, CoseKey, CoseSign1, CoseSign1Builder,
    Header, HeaderBuilder, KeyType, Label, ProtectedHeader,
};
use googletest::prelude::*;
use key_derivation::{HPKE_BASE_X25519_SHA256_AES128GCM, PUBLIC_KEY_CLAIM};
use key_management_service::{
    get_init_request, KeyManagementService, StorageKey, DST_NODE_IDS_CLAIM, INVOCATION_ID_CLAIM,
    LOGICAL_PIPELINE_NAME_CLAIM, TRANSFORM_INDEX_CLAIM,
};
use kms_proto::fcp::confidentialcompute::{
    authorize_confidential_transform_response::{AssociatedData, ProtectedResponse},
    key_management_service_server::KeyManagementService as _,
    keyset::Key,
    release_results_request::ReleasableResult,
    AuthorizeConfidentialTransformRequest, AuthorizeConfidentialTransformResponse,
    DeriveKeysRequest, DeriveKeysResponse, GetClusterPublicKeyRequest, GetKeysetRequest,
    GetLogicalPipelineStateRequest, Keyset, LogicalPipelineState,
    RegisterPipelineInvocationRequest, RegisterPipelineInvocationResponse, ReleaseResultsRequest,
    ReleaseResultsResponse, RotateKeysetRequest,
};
use oak_crypto::{encryptor::ServerEncryptor, signer::Signer};
use oak_proto_rust::oak::crypto::v1::Signature;
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use release_tokens::{
    ENCAPSULATED_KEY_PARAM, RELEASE_TOKEN_DST_STATE_PARAM, RELEASE_TOKEN_SRC_STATE_PARAM,
};
use session_test_utils::{
    get_test_encryption_key_handle, get_test_endorsements, get_test_evidence,
    get_test_reference_values, get_test_signer,
};
use storage::Storage;
use storage_client::StorageClient;
use storage_proto::{
    confidential_federated_compute::kms::{
        update_request, ClusterKeyValue, ReadRequest, ReadResponse, UpdateRequest, UpdateResponse,
    },
    duration_proto::google::protobuf::Duration,
    timestamp_proto::google::protobuf::Timestamp,
};
use tonic::{Code, IntoRequest, Response, Status};

#[derive(Default)]
struct FakeStorageClient {
    now: Arc<AtomicI64>,
    inner: tokio::sync::Mutex<Storage>,
}
impl FakeStorageClient {
    fn new(now: Arc<AtomicI64>) -> Self {
        Self { now, ..Default::default() }
    }
}
impl StorageClient for FakeStorageClient {
    async fn read(&self, request: ReadRequest) -> anyhow::Result<ReadResponse> {
        self.inner.lock().await.read(&request)
    }

    async fn update(&self, request: UpdateRequest) -> anyhow::Result<UpdateResponse> {
        self.inner.lock().await.update(
            &Timestamp { seconds: self.now.load(Ordering::Relaxed), ..Default::default() },
            request,
        )
    }
}

struct FakeSigner {}
#[async_trait::async_trait]
impl oak_sdk_containers::Signer for FakeSigner {
    async fn sign(&self, _data: &[u8]) -> anyhow::Result<Signature> {
        Ok(Signature { signature: b"signature".into() })
    }
}

/// Verifies that a HPKE private key and public key form a pair.
fn verify_hpke_keypair(private_key: &[u8], public_key: &[u8]) -> Result<()> {
    let private_key = CoseKey::from_slice(private_key);
    verify_that!(
        private_key,
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::OKP)),
            alg: some(eq(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))),
            key_id: not(empty()),
            params: contains_each![
                (
                    eq(Label::Int(iana::OkpKeyParameter::Crv as i64)),
                    eq(Value::from(iana::EllipticCurve::X25519 as u64)),
                ),
                (
                    eq(Label::Int(iana::OkpKeyParameter::D as i64)),
                    matches_pattern!(Value::Bytes(not(empty()))),
                ),
            ],
        }))
    )?;
    let raw_private_key = private_key
        .as_ref()
        .unwrap()
        .params
        .iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::D as i64))
        .map(|(_, value)| value.as_bytes().unwrap())
        .unwrap();

    let public_key = CoseKey::from_slice(public_key);
    verify_that!(
        public_key,
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::OKP)),
            alg: some(eq(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))),
            key_id: eq(private_key.as_ref().unwrap().key_id.as_slice()),
            params: contains_each![
                (
                    eq(Label::Int(iana::OkpKeyParameter::Crv as i64)),
                    eq(Value::from(iana::EllipticCurve::X25519 as u64)),
                ),
                (
                    eq(Label::Int(iana::OkpKeyParameter::X as i64)),
                    matches_pattern!(Value::Bytes(not(empty()))),
                ),
            ],
        }))
    )?;
    let raw_public_key = public_key
        .unwrap()
        .params
        .into_iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
        .map(|(_, value)| value.into_bytes().unwrap())
        .unwrap();

    verify_that!(
        hpke::Kem::X25519HkdfSha256.public_from_private(raw_private_key),
        some(eq(raw_public_key))
    )
}

#[googletest::test]
fn init_request() {
    let init_request = get_init_request();
    assert_that!(
        init_request,
        matches_pattern!(UpdateRequest {
            updates: elements_are![matches_pattern!(update_request::Update {
                key: eq(Vec::<u8>::try_from(StorageKey::ClusterKey).unwrap()),
                value: some(anything()),
                preconditions: elements_are![matches_pattern!(update_request::Preconditions {
                    exists: some(eq(false)),
                })],
            })],
        })
    );
    assert_that!(
        ClusterKeyValue::decode(init_request.updates[0].value.as_ref().unwrap().as_slice()),
        ok(matches_pattern!(ClusterKeyValue { key: not(empty()) }))
    );
}

#[googletest::test]
#[tokio::test]
async fn get_cluster_key() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let request = GetClusterPublicKeyRequest::default();
    let response = kms.get_cluster_public_key(request.into_request()).await;
    assert_that!(response, ok(anything()));
    let public_key = CoseKey::from_slice(response.unwrap().into_inner().public_key.as_slice());
    expect_that!(
        public_key,
        ok(matches_pattern!(CoseKey {
            kty: eq(KeyType::Assigned(iana::KeyType::EC2)),
            alg: some(eq(Algorithm::Assigned(iana::Algorithm::ES256))),
            key_id: not(empty()),
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
}

#[googletest::test]
#[tokio::test]
async fn get_keyset_with_empty_keyset() {
    let kms = KeyManagementService::new(FakeStorageClient::default(), FakeSigner {});

    let request = GetKeysetRequest { keyset_id: 1234 };
    let response = kms.get_keyset(request.into_request()).await;
    expect_that!(
        response,
        err(all!(
            property!(Status.code(), eq(Code::NotFound)),
            displays_as(contains_substring("no keys found in keyset")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn rotate_and_get_keyset() {
    let now = Arc::new(AtomicI64::new(1000));
    let kms = KeyManagementService::new(FakeStorageClient::new(now.clone()), FakeSigner {});
    // Add 3 keys, each with creation time (1000 + 10*i) and ttl 100*i.
    for i in 1..4 {
        now.fetch_add(10, Ordering::Relaxed);
        let request = RotateKeysetRequest {
            keyset_id: 1234,
            ttl: Some(Duration { seconds: 100 * i, nanos: 0 }),
        };
        let response = kms.rotate_keyset(request.into_request()).await;
        assert_that!(response, ok(anything()));
    }

    let request = GetKeysetRequest { keyset_id: 1234 };
    let response = kms.get_keyset(request.into_request()).await.map(Response::into_inner);
    assert_that!(
        response,
        ok(matches_pattern!(Keyset {
            keyset_id: eq(1234),
            keys: unordered_elements_are![
                matches_pattern!(Key {
                    key_id: not(empty()),
                    created: some(matches_pattern!(Timestamp { seconds: eq(1010) })),
                    expiration: some(matches_pattern!(Timestamp { seconds: eq(1110) })),
                }),
                matches_pattern!(Key {
                    key_id: not(empty()),
                    created: some(matches_pattern!(Timestamp { seconds: eq(1020) })),
                    expiration: some(matches_pattern!(Timestamp { seconds: eq(1220) })),
                }),
                matches_pattern!(Key {
                    key_id: not(empty()),
                    created: some(matches_pattern!(Timestamp { seconds: eq(1030) })),
                    expiration: some(matches_pattern!(Timestamp { seconds: eq(1330) })),
                }),
            ],
        }))
    );
    // The key ids should be unique.
    expect_that!(
        response.as_ref().unwrap().keys[0].key_id.as_slice(),
        not(eq(response.as_ref().unwrap().keys[1].key_id.as_slice()))
    );
}

#[googletest::test]
#[tokio::test]
async fn derive_keys_with_empty_keyset() {
    let kms = KeyManagementService::new(FakeStorageClient::default(), FakeSigner {});

    let request = DeriveKeysRequest {
        keyset_id: 1234,
        authorized_logical_pipeline_policies_hashes: vec![b"foo".into(), b"bar".into()],
    };
    let response = kms.derive_keys(request.into_request()).await;
    expect_that!(
        response,
        err(all!(
            property!(Status.code(), eq(Code::NotFound)),
            displays_as(contains_substring("no keys found in keyset")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn rotate_and_derive_keys() {
    let now = Arc::new(AtomicI64::new(1000));
    let kms = KeyManagementService::new(FakeStorageClient::new(now.clone()), FakeSigner {});
    // Add 3 keys, each with creation time (1000 + 10*i) and ttl 400 - 100*i.
    for i in 1..4 {
        now.fetch_add(10, Ordering::Relaxed);
        let request = RotateKeysetRequest {
            keyset_id: 1234,
            ttl: Some(Duration { seconds: 100 * (4 - i), nanos: 0 }),
        };
        let response = kms.rotate_keyset(request.into_request()).await;
        assert_that!(response, ok(anything()));
    }
    // Advance the clock one more time to 1050. A storage mutation is required
    // to update the stored current time, so we rotate a different keyset.
    now.fetch_add(20, Ordering::Relaxed);
    assert_that!(
        kms.rotate_keyset(
            RotateKeysetRequest { keyset_id: 0, ttl: Some(Duration { seconds: 10, nanos: 0 }) }
                .into_request()
        )
        .await,
        ok(anything()),
    );

    let request = DeriveKeysRequest {
        keyset_id: 1234,
        authorized_logical_pipeline_policies_hashes: vec![b"foo".into(), b"bar".into()],
    };
    let response = kms.derive_keys(request.into_request()).await.map(Response::into_inner);
    assert_that!(
        response,
        ok(matches_pattern!(DeriveKeysResponse {
            public_keys: elements_are![not(empty()), not(empty())],
        }))
    );

    let mut key_ids = vec![];
    for public_key in response.unwrap().public_keys {
        let cwt = CoseSign1::from_slice(&public_key);
        expect_that!(
            cwt,
            ok(matches_pattern!(CoseSign1 {
                protected: matches_pattern!(ProtectedHeader {
                    header: matches_pattern!(Header {
                        alg: some(eq(Algorithm::Assigned(iana::Algorithm::ES256))),
                    }),
                }),
                payload: some(anything()),
                signature: eq(b"signature"),
            }))
        );
        let claims = cwt.and_then(|cwt| ClaimsSet::from_slice(&cwt.payload.unwrap()));
        expect_that!(
            claims,
            ok(matches_pattern!(ClaimsSet {
                issued_at: some(eq(CwtTimestamp::WholeSeconds(1050))),
                // The most recently created key should be used, even if it
                // doesn't have the latest expiration.
                not_before: some(eq(CwtTimestamp::WholeSeconds(1030))),
                expiration_time: some(eq(CwtTimestamp::WholeSeconds(1130))),
                rest: contains((
                    eq(ClaimName::PrivateUse(PUBLIC_KEY_CLAIM)),
                    matches_pattern!(Value::Bytes(anything())),
                )),
            })),
        );
        let key = claims.and_then(|claims| {
            claims
                .rest
                .iter()
                .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
                .map(|(_, value)| CoseKey::from_slice(value.as_bytes().unwrap()))
                .unwrap()
        });
        expect_that!(
            key,
            ok(matches_pattern!(CoseKey {
                kty: eq(KeyType::Assigned(iana::KeyType::OKP)),
                alg: some(eq(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))),
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
        if let Ok(key_id) = key.map(|key| key.key_id) {
            key_ids.push(key_id);
        }
    }

    // The key ids should be distinct and start with the keyset key id.
    assert_that!(key_ids, len(eq(2)));
    expect_that!(&key_ids[0], not(eq(&key_ids[1])));
    let expected_prefix = kms
        .get_keyset(GetKeysetRequest { keyset_id: 1234 }.into_request())
        .await
        .map(Response::into_inner)
        .ok()
        .and_then(|response| {
            response
                .keys
                .into_iter()
                .find(|key| key.expiration.as_ref().map(|t| t.seconds) == Some(1130))
        })
        .map(|key| key.key_id)
        .expect("failed to find key with GetKeyset");
    expect_that!(key_ids[0], eq([expected_prefix.as_slice(), b"foo"].concat()));
    expect_that!(key_ids[1], eq([expected_prefix.as_slice(), b"bar"].concat()));
}

#[googletest::test]
#[tokio::test]
async fn get_logical_pipeline_state_with_empty_state() {
    let kms = KeyManagementService::new(FakeStorageClient::default(), FakeSigner {});

    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    let response = kms.get_logical_pipeline_state(request.into_request()).await;
    expect_that!(
        response,
        err(all!(
            property!(Status.code(), eq(Code::NotFound)),
            displays_as(contains_substring("logical pipeline has no saved state")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn register_pipeline_invocation_success() {
    let kms = KeyManagementService::new(FakeStorageClient::default(), FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src: 1,
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234, 5678],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response =
        kms.register_pipeline_invocation(request.into_request()).await.map(Response::into_inner);
    expect_that!(
        response,
        ok(matches_pattern!(RegisterPipelineInvocationResponse {
            invocation_id: not(empty()),
            logical_pipeline_state: none(), // There shouldn't be existing state.
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn register_pipeline_invocation_with_invalid_policies() {
    let kms = KeyManagementService::new(FakeStorageClient::default(), FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform { src: 1, ..Default::default() }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy {
                // Use a different policy that doesn't match `variant_policy`.
                instances: vec![PipelineVariantPolicy {
                    transforms: vec![Transform {
                        src: 2,
                        application: Some(ApplicationMatcher {
                            reference_values: Some(get_test_reference_values()),
                            ..Default::default()
                        }),

                        ..Default::default()
                    }],
                    ..Default::default()
                }],
            },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234, 5678],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response =
        kms.register_pipeline_invocation(request.into_request()).await.map(Response::into_inner);
    expect_that!(
        response,
        err(all!(
            property!(Status.code(), eq(Code::InvalidArgument)),
            displays_as(contains_substring("invalid pipeline policies")),
        ))
    );
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_success() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1, 2],
            dst_node_ids: vec![2, 3, 4],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            config_constraints: Some(Any { value: b"config".into(), ..Default::default() }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response = kms
        .register_pipeline_invocation(request.into_request())
        .await
        .expect("register_pipeline_invocation failed");

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: response.into_inner().invocation_id,
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    let response = kms
        .authorize_confidential_transform(request.into_request())
        .await
        .map(Response::into_inner);
    assert_that!(
        response,
        ok(matches_pattern!(AuthorizeConfidentialTransformResponse {
            protected_response: some(anything()),
        }))
    );
    let response = response.unwrap();

    let (_, plaintext, associated_data) = ServerEncryptor::decrypt_async(
        &response.protected_response.unwrap().convert().unwrap(),
        &get_test_encryption_key_handle(),
    )
    .await
    .expect("failed to decrypt response");

    // Verify the ProtectedResponse.
    expect_that!(
        ProtectedResponse::decode(plaintext.as_slice()),
        ok(matches_pattern!(ProtectedResponse {
            decryption_keys: len(eq(2)),
            result_encryption_keys: len(eq(3)),
        }))
    );

    // Verify the AssociatedData.
    let cluster_public_key = kms
        .get_cluster_public_key(GetClusterPublicKeyRequest::default().into_request())
        .await
        .expect("get_cluster_public_key failed");
    expect_that!(
        AssociatedData::decode(associated_data.as_slice()),
        ok(matches_pattern!(AssociatedData {
            cluster_public_key: eq(cluster_public_key.into_inner().public_key),
            config_constraints: some(matches_pattern!(Any { value: eq(b"config") })),
            authorized_logical_pipeline_policies_hashes: elements_are![eq(Sha256::hash(
                &logical_pipeline_policies.encode_to_vec()
            ))],
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_with_keyset_keys() {
    let now = Arc::new(AtomicI64::new(0));
    let storage_client = FakeStorageClient::new(now.clone());
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![0, 1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    }
    .encode_to_vec();

    // Add three keys to keyset 123, one of which expires before the pipeline
    // invocation intermediates. Add one key to keyset 456, and none to keyset
    // 789. Save the encryption keys for use later.
    let intermediates_exp = 100;
    let mut encryption_keys = Vec::new();
    for (keyset_id, exp) in [(123, 90), (123, 100), (123, 110), (456, 120)] {
        now.fetch_add(1, Ordering::Relaxed); // Avoid ambiguous creation times.
        let request = RotateKeysetRequest {
            keyset_id,
            ttl: Some(Duration { seconds: exp - now.load(Ordering::Relaxed), nanos: 0 }),
        };
        kms.rotate_keyset(request.into_request()).await.expect("rotate_keyset failed");
        if exp >= intermediates_exp {
            let request = DeriveKeysRequest {
                keyset_id,
                authorized_logical_pipeline_policies_hashes: vec![Sha256::hash(
                    &logical_pipeline_policies,
                )
                .into()],
            };
            let response =
                kms.derive_keys(request.into_request()).await.expect("derive_keys failed");
            let cwt =
                CoseSign1::from_slice(response.into_inner().public_keys.first().unwrap()).unwrap();
            let cose_key = ClaimsSet::from_slice(&cwt.payload.unwrap())
                .unwrap()
                .rest
                .into_iter()
                .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
                .and_then(|(_, value)| value.into_bytes().ok())
                .unwrap();
            encryption_keys.push((CoseKey::from_slice(&cose_key).unwrap().key_id, cose_key));
        }
    }

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration {
            seconds: intermediates_exp - now.load(Ordering::Relaxed),
            nanos: 0,
        }),
        keyset_ids: vec![123, 456, 789],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.clone()],
    };
    let response = kms
        .register_pipeline_invocation(request.into_request())
        .await
        .expect("register_pipeline_invocation failed");

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: response.into_inner().invocation_id,
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    let response = kms
        .authorize_confidential_transform(request.into_request())
        .await
        .expect("authorize_confidential_transform failed")
        .into_inner();

    let (_, plaintext, _) = ServerEncryptor::decrypt_async(
        &response.protected_response.unwrap().convert().unwrap(),
        &get_test_encryption_key_handle(),
    )
    .await
    .expect("failed to decrypt response");

    // Verify the ProtectedResponse. There should be 2 decryption keys for
    // keyset 123, 1 for keyset 456, and 1 for src_node_id 1.
    let protected_response = ProtectedResponse::decode(plaintext.as_slice())
        .expect("failed to decode ProtectedResponse");
    assert_that!(
        protected_response,
        matches_pattern!(ProtectedResponse {
            decryption_keys: len(eq(4)),
            result_encryption_keys: len(eq(1)),
        })
    );

    // For each public key that doesn't expire before the pipeline invocation,
    // there should be a corresponding decryption key.
    let decryption_keys = protected_response
        .decryption_keys
        .into_iter()
        .map(|key| (CoseKey::from_slice(&key).unwrap().key_id, key))
        .collect::<std::collections::HashMap<_, _>>();
    assert_that!(
        decryption_keys.keys().cloned().collect::<Vec<_>>(),
        superset_of(encryption_keys.iter().map(|(key_id, _)| key_id.clone()).collect::<Vec<_>>())
    );
    for (key_id, key) in encryption_keys {
        verify_hpke_keypair(decryption_keys.get(&key_id).unwrap(), &key).and_log_failure();
    }
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_with_intermediate_keys() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1, 2],
            dst_node_ids: vec![2, 3, 4],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response = kms
        .register_pipeline_invocation(request.into_request())
        .await
        .expect("register_pipeline_invocation failed");

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: response.into_inner().invocation_id,
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    let response = kms
        .authorize_confidential_transform(request.into_request())
        .await
        .expect("authorize_confidential_transform failed")
        .into_inner();

    let (_, plaintext, _) = ServerEncryptor::decrypt_async(
        &response.protected_response.unwrap().convert().unwrap(),
        &get_test_encryption_key_handle(),
    )
    .await
    .expect("failed to decrypt response");

    // Verify the ProtectedResponse.
    let protected_response = ProtectedResponse::decode(plaintext.as_slice())
        .expect("failed to decode ProtectedResponse");
    assert_that!(
        protected_response,
        matches_pattern!(ProtectedResponse {
            decryption_keys: len(eq(2)),
            result_encryption_keys: len(eq(3)),
        })
    );

    // Encryption and decryption keys with the same node id should be paired.
    verify_hpke_keypair(
        &protected_response.decryption_keys[1],
        &protected_response.result_encryption_keys[0],
    )
    .and_log_failure();
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_endorses_transform_signing_key() {
    let storage_client = FakeStorageClient::new(Arc::new(AtomicI64::new(1000)));
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1, 2],
            dst_node_ids: vec![2, 3, 4],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response = kms
        .register_pipeline_invocation(request.into_request())
        .await
        .expect("register_pipeline_invocation failed");
    let invocation_id = response.into_inner().invocation_id;

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: invocation_id.clone(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    let response = kms
        .authorize_confidential_transform(request.into_request())
        .await
        .expect("authorize_confidential_transform failed")
        .into_inner();

    // Convert the cluster public key into a bssl-crypto public key.
    let cluster_public_key = kms
        .get_cluster_public_key(GetClusterPublicKeyRequest::default().into_request())
        .await
        .expect("get_cluster_public_key failed");
    let public_key_params: std::collections::BTreeMap<_, _> =
        CoseKey::from_slice(&cluster_public_key.into_inner().public_key)
            .expect("failed to decode cluster public key")
            .params
            .into_iter()
            .collect();
    // X9.62 has the format "\x04<x><y>".
    let x962_public_key = [
        b"\x04",
        public_key_params
            .get(&Label::Int(iana::Ec2KeyParameter::X as i64))
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_slice(),
        public_key_params
            .get(&Label::Int(iana::Ec2KeyParameter::Y as i64))
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_slice(),
    ]
    .concat();
    let public_key = ecdsa::PublicKey::<ec::P256>::from_x962_uncompressed(&x962_public_key)
        .expect("invalid public key");

    // The signing key endorsement should be a valid CoseSign1 object signed by
    // the cluster key.
    let cwt = CoseSign1::from_slice(&response.signing_key_endorsement)
        .expect("failed to decode signing key endorsement");
    expect_that!(
        ClaimsSet::from_slice(cwt.payload.as_ref().unwrap()),
        ok(matches_pattern!(ClaimsSet {
            issued_at: some(eq(CwtTimestamp::WholeSeconds(1000))),
            not_before: some(eq(CwtTimestamp::WholeSeconds(1000))),
            expiration_time: some(eq(CwtTimestamp::WholeSeconds(1100))),
            rest: contains_each![
                (
                    eq(ClaimName::PrivateUse(LOGICAL_PIPELINE_NAME_CLAIM)),
                    matches_pattern!(Value::Text(eq("test"))),
                ),
                (
                    eq(ClaimName::PrivateUse(INVOCATION_ID_CLAIM)),
                    matches_pattern!(Value::Bytes(eq(invocation_id))),
                ),
                (eq(ClaimName::PrivateUse(TRANSFORM_INDEX_CLAIM)), eq(Value::from(0)),),
                (
                    eq(ClaimName::PrivateUse(DST_NODE_IDS_CLAIM)),
                    matches_pattern!(Value::Array(elements_are![
                        eq(Value::from(2)),
                        eq(Value::from(3)),
                        eq(Value::from(4)),
                    ])),
                ),
            ],
        }))
    );
    expect_that!(
        cwt.verify_signature(b"", |signature, data| public_key.verify_p1363(data, signature)),
        ok(anything())
    );
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_with_invalid_invocation_id() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: b"invalid".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    expect_that!(
        kms.authorize_confidential_transform(request.into_request()).await,
        err(displays_as(contains_substring("invocation_id is invalid")))
    );
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_without_registered_invocation() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: [0; 12].into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    expect_that!(
        kms.authorize_confidential_transform(request.into_request()).await,
        err(displays_as(contains_substring("pipeline invocation not found")))
    );
}

#[googletest::test]
#[tokio::test]
async fn authorize_confidential_transform_without_authorization() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                // These reference values won't match the test evidence.
                reference_values: Some(ReferenceValues::default()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![variant_policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    let request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: "test".into(),
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        intermediates_ttl: Some(Duration { seconds: 100, nanos: 0 }),
        keyset_ids: vec![1234],
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response = kms
        .register_pipeline_invocation(request.into_request())
        .await
        .expect("register_pipeline_invocation failed");

    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: response.into_inner().invocation_id,
        pipeline_variant_policy: variant_policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        tag: "tag".into(),
    };
    expect_that!(
        kms.authorize_confidential_transform(request.into_request()).await,
        err(displays_as(contains_substring("no transforms matched")))
    );
}

/// Helper function to register a pipeline invocation and authorize a transform.
/// Returns a tuple containing the current logical pipeline state,
/// ProtectedResponse containing encryption keys, and signing key endorsement.
///
/// This function is used to avoid repetitive (and complex) setup in tests
/// exercising the ReleaseResults flow.
async fn register_and_authorize<SC, S>(
    kms: &KeyManagementService<SC, S>,
    logical_pipeline_name: &str,
    policy: &PipelineVariantPolicy,
) -> anyhow::Result<(Option<LogicalPipelineState>, ProtectedResponse, Vec<u8>)>
where
    SC: StorageClient + Send + Sync + 'static,
    S: oak_sdk_containers::Signer + Send + Sync + 'static,
{
    register_and_authorize_with_options(
        kms,
        logical_pipeline_name,
        policy,
        Duration { seconds: 300, nanos: 0 },
        vec![],
    )
    .await
}

/// Variant of register_and_authorize that allows rarely-used options to be
/// specified.
async fn register_and_authorize_with_options<SC, S>(
    kms: &KeyManagementService<SC, S>,
    logical_pipeline_name: &str,
    policy: &PipelineVariantPolicy,
    intermediates_ttl: Duration,
    keyset_ids: Vec<u64>,
) -> anyhow::Result<(Option<LogicalPipelineState>, ProtectedResponse, Vec<u8>)>
where
    SC: StorageClient + Send + Sync + 'static,
    S: oak_sdk_containers::Signer + Send + Sync + 'static,
{
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            logical_pipeline_name.into(),
            LogicalPipelinePolicy { instances: vec![policy.clone()] },
        )]
        .into(),
        ..Default::default()
    };

    // Register the pipeline invocation.
    let register_invocation_request = RegisterPipelineInvocationRequest {
        logical_pipeline_name: logical_pipeline_name.into(),
        pipeline_variant_policy: policy.encode_to_vec(),
        intermediates_ttl: Some(intermediates_ttl),
        keyset_ids,
        authorized_logical_pipeline_policies: vec![logical_pipeline_policies.encode_to_vec()],
    };
    let response = kms
        .register_pipeline_invocation(register_invocation_request.into_request())
        .await
        .context("register_pipeline_invocation failed")?
        .into_inner();
    let logical_pipeline_state = response.logical_pipeline_state;

    // Authorize the transform.
    let request = AuthorizeConfidentialTransformRequest {
        invocation_id: response.invocation_id,
        pipeline_variant_policy: policy.encode_to_vec(),
        evidence: Some(get_test_evidence()),
        endorsements: Some(get_test_endorsements()),
        ..Default::default()
    };
    let response = kms
        .authorize_confidential_transform(request.into_request())
        .await
        .context("authorize_confidential_transform failed")?
        .into_inner();

    // Decrypt the ProtectedResponse.
    let (_, plaintext, _) = ServerEncryptor::decrypt_async(
        &response.protected_response.unwrap_or_default().convert().unwrap(),
        &get_test_encryption_key_handle(),
    )
    .await
    .context("failed to decrypt response")?;
    let protected_response = ProtectedResponse::decode(plaintext.as_slice())
        .context("failed to decode ProtectedResponse")?;

    Ok((logical_pipeline_state, protected_response, response.signing_key_endorsement))
}

/// Creates an encrypted release token with the given plaintext, source state,
/// and destination state.
fn create_release_token(
    plaintext: &[u8],
    src_state: Option<&[u8]>,
    dst_state: &[u8],
    cose_key: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let cose_key = CoseKey::from_slice(cose_key)
        .map_err(anyhow::Error::msg)
        .context("CoseKey::from_slice failed")?;
    let encryption_key = cose_key
        .params
        .into_iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
        .and_then(|(_, value)| value.into_bytes().ok())
        .context("public key missing")?;
    let (mut context, encapsulated_key) = hpke::SenderContext::new(
        &hpke::Params::new(
            hpke::Kem::X25519HkdfSha256,
            hpke::Kdf::HkdfSha256,
            hpke::Aead::Aes128Gcm,
        ),
        &encryption_key,
        b"",
    )
    .context("failed to create SenderContext")?;

    CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(
            CoseEncrypt0Builder::new()
                .protected(Header {
                    alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
                    rest: vec![
                        (
                            Label::Int(RELEASE_TOKEN_SRC_STATE_PARAM),
                            src_state.map(|s| Value::Bytes(s.into())).unwrap_or(Value::Null),
                        ),
                        (Label::Int(RELEASE_TOKEN_DST_STATE_PARAM), Value::Bytes(dst_state.into())),
                    ],
                    ..Default::default()
                })
                .unprotected(
                    HeaderBuilder::new()
                        .key_id(cose_key.key_id)
                        .value(ENCAPSULATED_KEY_PARAM, Value::Bytes(encapsulated_key))
                        .build(),
                )
                .create_ciphertext(plaintext, b"", move |plaintext, aad| {
                    context.seal(plaintext, aad)
                })
                .build()
                .to_vec()
                .map_err(anyhow::Error::msg)
                .context("failed to build CoseEncrypt0")?,
        )
        .create_signature(b"", |msg| get_test_signer().sign(msg))
        .build()
        .to_vec()
        .map_err(anyhow::Error::msg)
        .context("failed to build CoseSign1")
}

#[googletest::test]
#[tokio::test]
async fn release_results_success() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (logical_pipeline_state, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");
    expect_that!(logical_pipeline_state, none());

    // Call ReleaseResults to update the saved logical pipeline state.
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult {
            release_token,
            signing_key_endorsement: signing_key_endorsement.clone(),
        }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(ReleaseResultsResponse {
            decryption_keys: elements_are![eq(b"plaintext")],
            logical_pipeline_states: elements_are![matches_pattern!(LogicalPipelineState {
                name: eq("test"),
                value: eq(b"state")
            })],
        }))
    );

    // The logical pipeline state should now be updated. It should be returned
    // by subsequent calls to GetLogicalPipelineState and
    // RegisterPipelineInvocation.
    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    expect_that!(
        kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(LogicalPipelineState { name: eq("test"), value: eq(b"state") }))
    );
    expect_that!(
        register_and_authorize(&kms, "test", &variant_policy).await.map(|t| t.0),
        ok(some(matches_pattern!(LogicalPipelineState { name: eq("test"), value: eq(b"state") })))
    );

    // Another ReleaseResults call should update the state again. This verifies
    // that an update with initial state produces the right precondition.
    let release_token = create_release_token(
        b"plaintext2",
        /* src_state= */ Some(b"state"),
        /* dst_state= */ b"state2",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    assert_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(ReleaseResultsResponse {
            decryption_keys: elements_are![eq(b"plaintext2")],
            logical_pipeline_states: elements_are![matches_pattern!(LogicalPipelineState {
                name: eq("test"),
                value: eq(b"state2")
            })],
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_sets_expiration_from_intermediates_ttl() {
    let storage_client = FakeStorageClient::new(Arc::new(AtomicI64::new(10_000)));
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            dst_node_ids: vec![1],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) = register_and_authorize_with_options(
        &kms,
        "test",
        &variant_policy,
        Duration { seconds: 500, nanos: 0 },
        vec![],
    )
    .await
    .expect("failed to register and authorize pipeline transform");

    // Call ReleaseResults to update the saved logical pipeline state.
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(ReleaseResultsResponse {
            logical_pipeline_states: elements_are![matches_pattern!(LogicalPipelineState {
                expiration: some(matches_pattern!(Timestamp { seconds: eq(10_500) })),
            })],
        }))
    );

    // The expiration time should also be set in the GetLogicalPipelineState
    // response.
    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    expect_that!(
        kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(LogicalPipelineState {
            expiration: some(matches_pattern!(Timestamp { seconds: eq(10_500) }))
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_sets_expiration_from_key_ttl() {
    let now = Arc::new(AtomicI64::new(10_000));
    let storage_client = FakeStorageClient::new(now.clone());
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    // Create keys with different expiration times:
    //  - keyset 111: 20_000 (not used by pipeline)
    //  - keyset 222: 10_110, 10_220, 10_330
    //  - keyset 333: 10_240, 10_450
    let request =
        RotateKeysetRequest { keyset_id: 111, ttl: Some(Duration { seconds: 10_000, nanos: 0 }) };
    assert_that!(kms.rotate_keyset(request.into_request()).await, ok(anything()));
    for i in 1..4 {
        now.fetch_add(10, Ordering::Relaxed);
        let request = RotateKeysetRequest {
            keyset_id: 222,
            ttl: Some(Duration { seconds: 100 * i, nanos: 0 }),
        };
        assert_that!(kms.rotate_keyset(request.into_request()).await, ok(anything()));
    }
    for i in 1..3 {
        now.fetch_add(10, Ordering::Relaxed);
        let request = RotateKeysetRequest {
            keyset_id: 333,
            ttl: Some(Duration { seconds: 200 * i, nanos: 0 }),
        };
        assert_that!(kms.rotate_keyset(request.into_request()).await, ok(anything()));
    }

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            dst_node_ids: vec![1],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) = register_and_authorize_with_options(
        &kms,
        "test",
        &variant_policy,
        Duration { seconds: 100, nanos: 0 },
        vec![222, 333],
    )
    .await
    .expect("failed to register and authorize pipeline transform");

    // Call ReleaseResults to update the saved logical pipeline state. The
    // expiration time should be the larger of the key expiration times above
    // and the intermediates expiration time (10_150).
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(ReleaseResultsResponse {
            logical_pipeline_states: elements_are![matches_pattern!(LogicalPipelineState {
                expiration: some(matches_pattern!(Timestamp { seconds: eq(10_450) })),
            })],
        }))
    );

    // The expiration time should also be set in the GetLogicalPipelineState
    // response.
    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    expect_that!(
        kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(LogicalPipelineState {
            expiration: some(matches_pattern!(Timestamp { seconds: eq(10_450) }))
        }))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_sets_expiration_from_existing_logical_pipeline_state() {
    let storage_client = FakeStorageClient::new(Arc::new(AtomicI64::new(10_000)));
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            dst_node_ids: vec![1],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };

    for (src_state, dst_state, invocation_ttl) in
        [(None, b"state1", 500), (Some(b"state1".as_slice()), b"state2", 300)]
    {
        let (_, protected_response, signing_key_endorsement) = register_and_authorize_with_options(
            &kms,
            "test",
            &variant_policy,
            Duration { seconds: invocation_ttl, nanos: 0 },
            vec![],
        )
        .await
        .expect("failed to register and authorize pipeline transform");

        // Call ReleaseResults to update the saved logical pipeline state. Even
        // though the second invocation has a shorter TTL than the first, the
        // logical pipeline state's expiration time should not be shortened.
        let release_token = create_release_token(
            b"plaintext",
            src_state,
            dst_state,
            &protected_response.result_encryption_keys[0],
        )
        .expect("failed to create release token");
        let request = ReleaseResultsRequest {
            releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
        };
        expect_that!(
            kms.release_results(request.into_request()).await.map(Response::into_inner),
            ok(matches_pattern!(ReleaseResultsResponse {
                logical_pipeline_states: elements_are![matches_pattern!(LogicalPipelineState {
                    expiration: some(matches_pattern!(Timestamp { seconds: eq(10_500) })),
                })],
            }))
        );

        // The expiration time should also be set in the GetLogicalPipelineState
        // response.
        let request = GetLogicalPipelineStateRequest { name: "test".into() };
        expect_that!(
            kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
            ok(matches_pattern!(LogicalPipelineState {
                expiration: some(matches_pattern!(Timestamp { seconds: eq(10_500) }))
            }))
        );
    }
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_src_state_not_matching() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // ReleaseResults should fail if the initial state doesn't match the current
    // value (None).
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ Some(b"doesn't match"),
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_substring("failed to update logical pipeline states")))
    );

    // The logical pipeline state should not have been set.
    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    expect_that!(
        kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
        err(property!(Status.code(), eq(Code::NotFound)))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_src_state_none_not_matching() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // Set the logical pipeline state to "initial".
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"initial",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult {
            release_token,
            signing_key_endorsement: signing_key_endorsement.clone(),
        }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        ok(anything())
    );

    // ReleaseResults with an initial state that doesn't match the current
    // value ("initial").
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_substring("failed to update logical pipeline states")))
    );

    // The logical pipeline state should not have been changed.
    let request = GetLogicalPipelineStateRequest { name: "test".into() };
    expect_that!(
        kms.get_logical_pipeline_state(request.into_request()).await.map(Response::into_inner),
        ok(matches_pattern!(LogicalPipelineState { value: eq(b"initial") }))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_missing_invocation_id() {
    let now = Arc::new(AtomicI64::new(1000));
    let storage_client = FakeStorageClient::new(now.clone());
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // Advance the clock and trigger a storage update to ensure that the
    // invocation expires.
    now.fetch_add(3600, Ordering::Relaxed);
    register_and_authorize(&kms, "test", &variant_policy)
        .await
        .expect("failed to register and authorize pipeline transform");

    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_regex("pipeline invocation .+ not found")))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_modified_release_token() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // Tamper with the release token.
    let mut release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    for i in 0..release_token.len() {
        if release_token[i..].starts_with(b"state") {
            release_token[i..i + 5].copy_from_slice(b"other");
            break;
        }
    }

    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_substring("releasable_results are invalid")))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_undecryptable_release_token() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // Create a release token that cannot be decrypted.
    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let mut token = CoseSign1::from_slice(&release_token).unwrap();
    token.payload.as_mut().unwrap()[0] ^= 0xFF;
    let release_token = token.to_vec().unwrap();

    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_substring("releasable_results are invalid")))
    );
}

#[googletest::test]
#[tokio::test]
async fn release_results_fails_with_modified_signing_key_endorsement() {
    let storage_client = FakeStorageClient::default();
    assert_that!(storage_client.update(get_init_request()).await, ok(anything()));
    let kms = KeyManagementService::new(storage_client, FakeSigner {});

    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![1],
            dst_node_ids: vec![2],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (_, protected_response, signing_key_endorsement) =
        register_and_authorize(&kms, "test", &variant_policy)
            .await
            .expect("failed to register and authorize pipeline transform");

    // Tamper with the signing key endorsement. Modifications to the signature
    // won't be caught until later in the flow.
    let mut cwt = CoseSign1::from_slice(&signing_key_endorsement)
        .expect("failed to parse signing key endorsement");
    cwt.signature[0] ^= 0xFF;
    let signing_key_endorsement = cwt.to_vec().unwrap();

    let release_token = create_release_token(
        b"plaintext",
        /* src_state= */ None,
        /* dst_state= */ b"state",
        &protected_response.result_encryption_keys[0],
    )
    .expect("failed to create release token");
    let request = ReleaseResultsRequest {
        releasable_results: vec![ReleasableResult { release_token, signing_key_endorsement }],
    };
    expect_that!(
        kms.release_results(request.into_request()).await.map(Response::into_inner),
        err(displays_as(contains_substring("releasable_results are invalid")))
    );
}

mod storage_key {
    use googletest::prelude::*;
    use key_management_service::StorageKey;

    #[googletest::test]
    fn round_trip_cluster_key() {
        let storage_key = StorageKey::ClusterKey;
        let encoded: anyhow::Result<Vec<u8>> = storage_key.try_into();
        assert_that!(encoded, ok(anything()));
        expect_that!(StorageKey::try_from(encoded.unwrap().as_slice()), ok(eq(storage_key)));
    }

    #[googletest::test]
    fn round_trip_keyset_key() {
        let storage_key = StorageKey::KeysetKey { keyset_id: 1234, key_id: *b"abcd" };
        let encoded: anyhow::Result<Vec<u8>> = storage_key.try_into();
        assert_that!(encoded, ok(anything()));
        expect_that!(StorageKey::try_from(encoded.unwrap().as_slice()), ok(eq(storage_key)));
    }

    #[googletest::test]
    fn round_trip_pipeline_invocation_state() {
        let storage_key = StorageKey::PipelineInvocationState { id: *b"0123456789ab" };
        let encoded: anyhow::Result<Vec<u8>> = storage_key.try_into();
        assert_that!(encoded, ok(anything()));
        expect_that!(StorageKey::try_from(encoded.unwrap().as_slice()), ok(eq(storage_key)));
    }

    #[googletest::test]
    fn round_trip_logical_pipeline_state() {
        let storage_key = StorageKey::LogicalPipelineState { id: *b"\x80123456789abcdef" };
        let encoded: anyhow::Result<Vec<u8>> = storage_key.try_into();
        assert_that!(encoded, ok(anything()));
        expect_that!(StorageKey::try_from(encoded.unwrap().as_slice()), ok(eq(storage_key)));
    }

    #[googletest::test]
    fn encode_invalid_logical_pipeline_state() {
        let storage_key = StorageKey::LogicalPipelineState { id: *b"0123456789abcdef" };
        let encoded: anyhow::Result<Vec<u8>> = storage_key.try_into();
        expect_that!(encoded, err(displays_as(contains_substring("invalid LogicalPipeline id"))));
    }

    #[googletest::test]
    fn decode_invalid_keyset_key() {
        // The encoding must be 16 bytes.
        expect_that!(
            StorageKey::try_from([0u8; 0].as_slice()),
            err(displays_as(contains_substring("invalid StorageKey length")))
        );
        expect_that!(
            StorageKey::try_from([0u8; 15].as_slice()),
            err(displays_as(contains_substring("invalid StorageKey length")))
        );
        expect_that!(
            StorageKey::try_from([0u8; 17].as_slice()),
            err(displays_as(contains_substring("invalid StorageKey length")))
        );
    }

    #[googletest::test]
    fn decode_reserved_keyset_key_encoding() {
        // If the first 4 bytes are 0, the last 12 must also be 0.
        expect_that!(
            StorageKey::try_from(0x00000000000000000000000000000001u128.to_be_bytes().as_slice()),
            err(displays_as(contains_substring("unsupported StorageKey encoding")))
        );

        // Initial u32 values in [0x00000003, 0x7FFFFFFF] are reserved.
        expect_that!(
            StorageKey::try_from(0x00000003000000000000000000000000u128.to_be_bytes().as_slice()),
            err(displays_as(contains_substring("unsupported StorageKey encoding")))
        );
        expect_that!(
            StorageKey::try_from(0x11111111000000000000000000000000u128.to_be_bytes().as_slice()),
            err(displays_as(contains_substring("unsupported StorageKey encoding")))
        );
        expect_that!(
            StorageKey::try_from(0x66666666000000000000000000000000u128.to_be_bytes().as_slice()),
            err(displays_as(contains_substring("unsupported StorageKey encoding")))
        );
        expect_that!(
            StorageKey::try_from(0x7FFFFFFF000000000000000000000000u128.to_be_bytes().as_slice()),
            err(displays_as(contains_substring("unsupported StorageKey encoding")))
        );
    }
}
