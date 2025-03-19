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

use coset::{cbor::value::Value, iana, Algorithm, CborSerializable, CoseKey, KeyType, Label};
use googletest::prelude::*;
use key_management_service::{get_init_request, KeyManagementService, StorageKey};
use kms_proto::fcp::confidentialcompute::{
    key_management_service_server::KeyManagementService as _, GetClusterPublicKeyRequest,
};
use prost::Message;
use storage::Storage;
use storage_client::StorageClient;
use storage_proto::confidential_federated_compute::kms::{
    update_request, ClusterKeyValue, ReadRequest, ReadResponse, UpdateRequest, UpdateResponse,
};
use tonic::IntoRequest;

#[derive(Default)]
struct FakeStorageClient {
    inner: tokio::sync::Mutex<Storage>,
}
impl StorageClient for FakeStorageClient {
    async fn read(&self, request: ReadRequest) -> anyhow::Result<ReadResponse> {
        self.inner.lock().await.read(&request)
    }

    async fn update(&self, request: UpdateRequest) -> anyhow::Result<UpdateResponse> {
        self.inner.lock().await.update(&Default::default(), request)
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
    let kms = KeyManagementService::new(storage_client);

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
