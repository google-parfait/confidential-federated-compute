// Copyright 2026 Google LLC.
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
#include "containers/fns/fn_utils.h"

#include <string>

#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

TEST(FnUtilsTest, GetBlobIdUnencryptedReturnsBlobId) {
  BlobMetadata metadata;
  metadata.mutable_unencrypted()->set_blob_id("test_blob_id");
  EXPECT_EQ(GetBlobId(metadata), "test_blob_id");
}

TEST(FnUtilsTest, GetBlobIdHpkePlusAeadDataReturnsBlobId) {
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()->set_blob_id("another_blob_id");
  EXPECT_EQ(GetBlobId(metadata), "another_blob_id");
}

TEST(FnUtilsTest, GetBlobIdEmptyMetadataReturnsEmptyString) {
  BlobMetadata metadata;
  EXPECT_EQ(GetBlobId(metadata), "");
}

TEST(ExtractAssociatedMetadataTest,
     ExtractsMetadataFromHpkeWithKmsAssociatedData) {
  BlobHeader header;
  header.set_blob_id("test_blob");
  AssociatedMetadata assoc_metadata;
  assoc_metadata.add_metadata()->PackFrom(header);
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(assoc_metadata);

  AssociatedMetadata result = ExtractAssociatedMetadata(metadata);
  ASSERT_EQ(result.metadata_size(), 1);
  BlobHeader unpacked;
  EXPECT_TRUE(result.metadata(0).UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "test_blob");
}

}  // namespace
}  // namespace confidential_federated_compute::fns
