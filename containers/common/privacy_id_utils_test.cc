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

#include "containers/common/privacy_id_utils.h"

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::PrivacyIdMetadata;
using ::fcp::confidentialcompute::ProtectedMetadata;
using ::testing::Optional;

// Appends a PrivacyIdMetadata entry with the given privacy ID.
void AddPrivacyIdMetadata(absl::string_view privacy_id,
                          ProtectedMetadata& protected_metadata) {
  PrivacyIdMetadata privacy_id_metadata;
  privacy_id_metadata.set_privacy_id(std::string(privacy_id));
  protected_metadata.add_metadata()->PackFrom(privacy_id_metadata);
}

// Appends an entry of a type other than PrivacyIdMetadata.
void AddOtherMetadata(ProtectedMetadata& protected_metadata) {
  BlobHeader blob_header;
  blob_header.set_blob_id("blob_id");
  protected_metadata.add_metadata()->PackFrom(blob_header);
}

TEST(GetPrivacyIdFromProtectedMetadataTest, ReturnsPrivacyId) {
  ProtectedMetadata protected_metadata;
  AddPrivacyIdMetadata("privacy_id", protected_metadata);

  EXPECT_THAT(GetPrivacyIdFromProtectedMetadata(protected_metadata),
              Optional(std::string("privacy_id")));
}

TEST(GetPrivacyIdFromProtectedMetadataTest, ReturnsEmptyPrivacyIdIfSet) {
  ProtectedMetadata protected_metadata;
  AddPrivacyIdMetadata("", protected_metadata);

  EXPECT_THAT(GetPrivacyIdFromProtectedMetadata(protected_metadata),
              Optional(std::string("")));
}

TEST(GetPrivacyIdFromProtectedMetadataTest, SkipsOtherEntries) {
  ProtectedMetadata protected_metadata;
  AddOtherMetadata(protected_metadata);
  AddPrivacyIdMetadata("privacy_id", protected_metadata);
  AddOtherMetadata(protected_metadata);

  EXPECT_THAT(GetPrivacyIdFromProtectedMetadata(protected_metadata),
              Optional(std::string("privacy_id")));
}

TEST(GetPrivacyIdFromProtectedMetadataTest, ReturnsFirstPrivacyId) {
  ProtectedMetadata protected_metadata;
  AddPrivacyIdMetadata("privacy_id_1", protected_metadata);
  AddPrivacyIdMetadata("privacy_id_2", protected_metadata);

  EXPECT_THAT(GetPrivacyIdFromProtectedMetadata(protected_metadata),
              Optional(std::string("privacy_id_1")));
}

TEST(GetPrivacyIdFromProtectedMetadataTest, ReturnsNulloptIfEmpty) {
  EXPECT_EQ(GetPrivacyIdFromProtectedMetadata(ProtectedMetadata()),
            std::nullopt);
}

TEST(GetPrivacyIdFromProtectedMetadataTest,
     ReturnsNulloptWithoutPrivacyIdMetadata) {
  ProtectedMetadata protected_metadata;
  AddOtherMetadata(protected_metadata);

  EXPECT_EQ(GetPrivacyIdFromProtectedMetadata(protected_metadata),
            std::nullopt);
}

TEST(GetPrivacyIdFromProtectedMetadataTest, SkipsMalformedPrivacyIdMetadata) {
  ProtectedMetadata protected_metadata;
  google::protobuf::Any* malformed_entry = protected_metadata.add_metadata();
  malformed_entry->PackFrom(PrivacyIdMetadata());
  // A truncated varint, which can't be parsed as a PrivacyIdMetadata.
  malformed_entry->set_value("\xff");

  EXPECT_EQ(GetPrivacyIdFromProtectedMetadata(protected_metadata),
            std::nullopt);

  AddPrivacyIdMetadata("privacy_id", protected_metadata);
  EXPECT_THAT(GetPrivacyIdFromProtectedMetadata(protected_metadata),
              Optional(std::string("privacy_id")));
}

}  // namespace
}  // namespace confidential_federated_compute
