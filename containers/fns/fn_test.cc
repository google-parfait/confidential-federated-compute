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
#include "containers/fns/fn.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::StrictMock;

class MockFn : public Fn {
 public:
  MOCK_METHOD(absl::Status, InitializeReplica,
              (Any config, ConfigureContext& context), (override));
  MOCK_METHOD(absl::Status, FinalizeReplica, (Any config, FnContext& context),
              (override));
  MOCK_METHOD(absl::StatusOr<WriteFinishedResponse>, Write,
              (WriteRequest write_request, std::string unencrypted_data,
               Context& context),
              (override));
  MOCK_METHOD(absl::StatusOr<CommitResponse>, Commit,
              (CommitRequest commit_request, Context& context), (override));
};

AssociatedMetadata CreateMetadata(const std::string& blob_id,
                                  const std::string& key_id) {
  BlobHeader header;
  header.set_blob_id(blob_id);
  header.set_key_id(key_id);
  AssociatedMetadata metadata;
  metadata.add_metadata()->PackFrom(header);
  return metadata;
}

class FnTest : public testing::Test {
 protected:
  StrictMock<MockContext> mock_context_;
};

TEST_F(FnTest, UnpackMetadataMatches) {
  BlobHeader header;
  header.set_blob_id("test_blob");
  header.set_key_id("test_key");
  AssociatedMetadata metadata;
  metadata.add_metadata()->PackFrom(header);
  Fn::FnContext fn_context(mock_context_, std::move(metadata));

  BlobHeader unpacked;
  ASSERT_TRUE(fn_context.UnpackMetadata(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "test_blob");
  EXPECT_EQ(unpacked.key_id(), "test_key");
}

TEST_F(FnTest, UnpackMetadataNotFound) {
  AssociatedMetadata metadata;
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  BlobHeader unpacked;
  EXPECT_FALSE(fn_context.UnpackMetadata(&unpacked));
}

TEST_F(FnTest, PackMetadataNewEntry) {
  AssociatedMetadata metadata;
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  BlobHeader header;
  header.set_blob_id("new_blob");
  fn_context.PackMetadata(header);

  BlobHeader unpacked;
  ASSERT_TRUE(fn_context.UnpackMetadata(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "new_blob");
}

TEST_F(FnTest, PackMetadataOverwritesExistingEntry) {
  AssociatedMetadata metadata = CreateMetadata("blob_id_1", "key_id_1");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  BlobHeader new_header;
  new_header.set_blob_id("blob_id_2");
  new_header.set_key_id("key_id_2");
  fn_context.PackMetadata(new_header);

  BlobHeader unpacked;
  ASSERT_TRUE(fn_context.UnpackMetadata(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "blob_id_2");
  EXPECT_EQ(unpacked.key_id(), "key_id_2");
}

TEST_F(FnTest, RemoveMetadataExplicitType) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  fn_context.RemoveMetadata<BlobHeader>();
  BlobHeader unpacked;
  EXPECT_FALSE(fn_context.UnpackMetadata(&unpacked));
}

TEST_F(FnTest, RemoveMetadataWithObject) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  fn_context.RemoveMetadata(BlobHeader{});
  BlobHeader unpacked;
  EXPECT_FALSE(fn_context.UnpackMetadata(&unpacked));
}

TEST_F(FnTest, ClearMetadata) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  fn_context.ClearMetadata();
  BlobHeader unpacked;
  EXPECT_FALSE(fn_context.UnpackMetadata(&unpacked));
}

TEST_F(FnTest, EmitEncryptedAttachesMetadataToKV) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  Session::KV emitted_kv;
  EXPECT_CALL(mock_context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv), Return(true)));

  Session::KV output;
  output.data = "output_data";
  EXPECT_TRUE(fn_context.EmitEncrypted(0, std::move(output)));
  ASSERT_TRUE(emitted_kv.associated_metadata.has_value());
  EXPECT_EQ(emitted_kv.associated_metadata->metadata_size(), 1);
  BlobHeader unpacked;
  EXPECT_TRUE(emitted_kv.associated_metadata->metadata(0).UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "blob_id");
}

TEST_F(FnTest, EmitEncryptedDoesNotOverrideExplicitMetadata) {
  AssociatedMetadata input_metadata =
      CreateMetadata("input_blob_id", "input_key_id");
  Fn::FnContext fn_context(mock_context_, std::move(input_metadata));
  AssociatedMetadata explicit_metadata =
      CreateMetadata("explicit_blob_id", "explicit_key_id");
  Session::KV emitted_kv;
  EXPECT_CALL(mock_context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv), Return(true)));

  Session::KV output;
  output.data = "output_data";
  output.associated_metadata = explicit_metadata;
  EXPECT_TRUE(fn_context.EmitEncrypted(0, std::move(output)));
  ASSERT_TRUE(emitted_kv.associated_metadata.has_value());
  BlobHeader unpacked;
  EXPECT_TRUE(emitted_kv.associated_metadata->metadata(0).UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.blob_id(), "explicit_blob_id");
}

TEST_F(FnTest, EmitUnencryptedAttachesMetadata) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  Session::KV emitted_kv;
  EXPECT_CALL(mock_context_, EmitUnencrypted(_))
      .WillOnce(DoAll(SaveArg<0>(&emitted_kv), Return(true)));

  Session::KV output;
  output.data = "output_data";
  EXPECT_TRUE(fn_context.EmitUnencrypted(std::move(output)));
  ASSERT_TRUE(emitted_kv.associated_metadata.has_value());
  EXPECT_EQ(emitted_kv.associated_metadata->metadata_size(), 1);
}

TEST_F(FnTest, EmitReleasableAttachesMetadata) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));

  Session::KV emitted_kv;
  EXPECT_CALL(mock_context_, EmitReleasable(0, _, _, _, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv),
                      testing::SetArgReferee<4>("test_release_token"),
                      Return(true)));
  Session::KV output;
  output.data = "output_data";
  EXPECT_TRUE(
      fn_context.EmitReleasable(0, std::move(output), std::nullopt, "dst"));
  ASSERT_TRUE(emitted_kv.associated_metadata.has_value());
  EXPECT_EQ(emitted_kv.associated_metadata->metadata_size(), 1);
}

TEST_F(FnTest, MultipleEmissionsGetSameMetadata) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));
  Session::KV emitted_kv1;
  Session::KV emitted_kv2;
  EXPECT_CALL(mock_context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv1), Return(true)))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv2), Return(true)));
  Session::KV output1;
  output1.data = "data1";
  EXPECT_TRUE(fn_context.EmitEncrypted(0, std::move(output1)));
  Session::KV output2;
  output2.data = "data2";
  EXPECT_TRUE(fn_context.EmitEncrypted(0, std::move(output2)));

  // Both emissions should have the same metadata.
  ASSERT_TRUE(emitted_kv1.associated_metadata.has_value());
  ASSERT_TRUE(emitted_kv2.associated_metadata.has_value());
  BlobHeader h1, h2;
  EXPECT_TRUE(emitted_kv1.associated_metadata->metadata(0).UnpackTo(&h1));
  EXPECT_TRUE(emitted_kv2.associated_metadata->metadata(0).UnpackTo(&h2));
  EXPECT_EQ(h1.blob_id(), "blob_id");
  EXPECT_EQ(h2.blob_id(), "blob_id");
}

TEST_F(FnTest, EmitReleasableReleaseTokenAlreadySet) {
  AssociatedMetadata metadata = CreateMetadata("blob_id", "key_id");
  Fn::FnContext fn_context(mock_context_, std::move(metadata));

  EXPECT_CALL(mock_context_, EmitReleasable(0, _, _, _, _))
      .WillOnce(
          DoAll(testing::SetArgReferee<4>("test_release_token"), Return(true)));
  Session::KV output1;
  output1.data = "output_data_1";
  EXPECT_TRUE(
      fn_context.EmitReleasable(0, std::move(output1), std::nullopt, "dst"));

  Session::KV output2;
  output2.data = "output_data_2";
  EXPECT_FALSE(
      fn_context.EmitReleasable(0, std::move(output2), std::nullopt, "dst"));
}

TEST_F(FnTest, FinalizeReturnsReleaseToken) {
  MockFn session;
  EXPECT_CALL(session, FinalizeReplica(_, _))
      .WillOnce([](Any, Fn::FnContext& ctx) -> absl::Status {
        Session::KV kv;
        kv.data = "released_data";
        ctx.EmitReleasable(0, std::move(kv), std::nullopt, "dst");
        return absl::OkStatus();
      });
  EXPECT_CALL(mock_context_, EmitReleasable(0, _, _, _, _))
      .WillOnce(
          DoAll(testing::SetArgReferee<4>("my_release_token"), Return(true)));

  FinalizeRequest finalize_request;
  BlobMetadata blob_metadata;
  auto finalize_result =
      session.Finalize(finalize_request, blob_metadata, mock_context_);
  ASSERT_THAT(finalize_result, IsOk());
  EXPECT_EQ(finalize_result->release_token(), "my_release_token");
}

TEST_F(FnTest, IncrementCounter) {
  EXPECT_CALL(mock_context_, GetCounters())
      .WillRepeatedly(testing::ReturnRef(mock_context_.counters_));

  AssociatedMetadata metadata;
  Fn::FnContext fn_context(mock_context_, std::move(metadata));

  fn_context.IncrementCounter("counter_a");
  fn_context.IncrementCounter("counter_a");
  fn_context.IncrementCounter("counter_a");
  fn_context.IncrementCounter("counter_b");
  fn_context.IncrementCounter("counter_b");

  EXPECT_EQ(mock_context_.counters_["counter_a"], 3);
  EXPECT_EQ(mock_context_.counters_["counter_b"], 2);
}

TEST_F(FnTest, IncrementCounterByAmount) {
  EXPECT_CALL(mock_context_, GetCounters())
      .WillRepeatedly(testing::ReturnRef(mock_context_.counters_));

  AssociatedMetadata metadata;
  Fn::FnContext fn_context(mock_context_, std::move(metadata));

  fn_context.IncrementCounterBy("my_counter_a", 10);
  fn_context.IncrementCounterBy("my_counter_a", 20);
  fn_context.IncrementCounterBy("my_counter_b", 5);
  fn_context.IncrementCounterBy("my_counter_b", 10);

  EXPECT_EQ(mock_context_.counters_["my_counter_a"], 30);
  EXPECT_EQ(mock_context_.counters_["my_counter_b"], 15);
}

}  // namespace
}  // namespace confidential_federated_compute::fns
