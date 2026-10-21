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

#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

BlobMetadata MakeEncryptedMetadata(absl::string_view blob_id_copy) {
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()->set_blob_id(
      std::string(blob_id_copy));
  return metadata;
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdUnencryptedReturnsBlobId) {
  BlobMetadata metadata;
  metadata.mutable_unencrypted()->set_blob_id("test_blob_id");
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "test_blob_id");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdReturnsDirectBlobHeaderBlobId) {
  BlobHeader header;
  header.set_blob_id("header_blob_id");
  header.set_key_id("key_id");
  BlobMetadata metadata = MakeEncryptedMetadata("header_blob_id");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(header);
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "header_blob_id");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdReturnsBundledBlobHeaderBlobId) {
  BlobHeader header;
  header.set_blob_id("header_blob_id");
  header.set_key_id("key_id");
  AssociatedMetadata associated_metadata;
  associated_metadata.add_metadata()->PackFrom(header);
  BlobMetadata metadata = MakeEncryptedMetadata("header_blob_id");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(associated_metadata);
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "header_blob_id");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdIgnoresUnauthenticatedBlobIdCopy) {
  BlobHeader header;
  header.set_blob_id("header_blob_id");
  header.set_key_id("key_id");
  BlobMetadata metadata = MakeEncryptedMetadata("other_blob_id");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(header);
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "header_blob_id");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdWithoutBlobHeaderReturnsEmptyString) {
  // E.g. sessionized inputs, whose associated data has no BlobHeader.
  AssociatedMetadata associated_metadata;
  BlobMetadata metadata = MakeEncryptedMetadata("blob_id_copy");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(associated_metadata);
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdWithoutAssociatedDataReturnsEmpty) {
  EXPECT_EQ(GetAuthenticatedBlobId(MakeEncryptedMetadata("blob_id_copy")), "");
}

TEST(FnUtilsTest, GetAuthenticatedBlobIdEmptyMetadataReturnsEmptyString) {
  BlobMetadata metadata;
  EXPECT_EQ(GetAuthenticatedBlobId(metadata), "");
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

TEST(ExtractAssociatedMetadataTest,
     ExtractsDirectBlobHeaderFromHpkeWithKmsAssociatedData) {
  BlobHeader header;
  header.set_blob_id("test_blob");
  header.set_key_id("test_key_id");
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->set_type_url("type.googleapis.com/fcp.confidentialcompute.BlobHeader");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->set_value(header.SerializeAsString());

  AssociatedMetadata result = ExtractAssociatedMetadata(metadata);
  ASSERT_EQ(result.metadata_size(), 1);
  BlobHeader unpacked;
  EXPECT_TRUE(result.metadata(0).UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "test_blob");
  EXPECT_EQ(unpacked.key_id(), "test_key_id");
}

}  // namespace
}  // namespace confidential_federated_compute::fns
