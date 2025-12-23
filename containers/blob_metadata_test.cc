// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "containers/blob_metadata.h"

#include "absl/status/status_matchers.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

TEST(GetKeyIdFromMetadata, SuccessUnencrypted) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();
  EXPECT_EQ(GetKeyIdFromMetadata(unencrypted).value(), "");
}

TEST(GetKeyIdFromMetadata, SuccessEncrypted) {
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id("key_id");
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(header.SerializeAsString());

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).value(), "key_id");
}

TEST(GetKeyIdFromMetadata, InvalidAssociatedData) {
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header("invalid!!!");

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(GetKeyIdFromMetadata, NoKmsAssociatedData) {
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key("some_key");

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace confidential_federated_compute
