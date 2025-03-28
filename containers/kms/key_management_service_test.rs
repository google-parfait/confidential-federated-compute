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

use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet, Timestamp as CwtTimestamp},
    iana, Algorithm, CborSerializable, CoseKey, CoseSign1, Header, KeyType, Label, ProtectedHeader,
};
use googletest::prelude::*;
use key_derivation::{HPKE_BASE_X25519_SHA256_AES128GCM, PUBLIC_KEY_CLAIM};
use key_management_service::{get_init_request, KeyManagementService, StorageKey};
use kms_proto::fcp::confidentialcompute::{
    key_management_service_server::KeyManagementService as _, keyset::Key, DeriveKeysRequest,
    DeriveKeysResponse, GetClusterPublicKeyRequest, GetKeysetRequest, Keyset, RotateKeysetRequest,
};
use oak_proto_rust::oak::crypto::v1::Signature;
use prost::Message;
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
        authorized_logical_pipelines_hashes: vec![b"foo".into(), b"bar".into()],
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
        authorized_logical_pipelines_hashes: vec![b"foo".into(), b"bar".into()],
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
