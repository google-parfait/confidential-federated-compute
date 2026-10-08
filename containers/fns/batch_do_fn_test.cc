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

#include "containers/fns/batch_do_fn.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::ProtectedMetadata;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::testing::_;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SizeIs;
using ::testing::StrictMock;
using Element = BatchDoFn::Element;
using DoContext = BatchDoFn::DoContext;

// Like SaveArg<k>, but unpacks the AssociatedMetadata packed into the Any
// argument passed to Session::Context::EmitEncrypted.
template <size_t k>
auto SaveAssociatedMetadataArg(AssociatedMetadata* metadata) {
  return testing::WithArg<k>(
      [metadata](const Any& any) { EXPECT_TRUE(any.UnpackTo(metadata)); });
}

// Mock Context for verifying emissions.
class MockContext : public Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
  MOCK_METHOD(bool, EmitUnencrypted, (Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV, google::protobuf::Any),
              (override));
  MOCK_METHOD(bool, EmitReleasable,
              (int, Session::KV, std::optional<absl::string_view>,
               absl::string_view, std::string&),
              (override));
  MOCK_METHOD(Counters&, GetCounters, (), (override));
};

// A concrete subclass for testing. Records the inputs received by Do()
// and can be configured to emit specific outputs or return errors.
class TestBatchDoFn : public BatchDoFn {
 public:
  absl::Status Do(Any config, std::vector<Element> inputs,
                  DoContext& context) override {
    if (custom_do_) {
      return custom_do_(std::move(config), std::move(inputs), context);
    }
    if (do_error_.has_value()) {
      return *do_error_;
    }

    // Emit a summary: concatenate all input data, keys, and blob_ids.
    std::string combined;
    for (const auto& input : inputs) {
      combined += input.kv.data;
    }
    Session::KV output;
    output.data = std::move(combined);
    // Encode config into the output so tests can verify it was forwarded.
    output.blob_id = config.type_url();
    // Encode input count into the key.
    Any count_key;
    count_key.set_value(std::to_string(inputs.size()));
    output.key = std::move(count_key);
    context.EmitUnencrypted(std::move(output));

    // Also emit each input individually so tests can verify key/blob_id.
    for (auto& input : inputs) {
      context.EmitUnencrypted(std::move(input.kv));
    }
    return absl::OkStatus();
  }

  // Configure Do() to return an error.
  void SetDoError(absl::Status error) { do_error_ = std::move(error); }

  // Configure a custom Do() implementation for specific test cases.
  void SetCustomDo(
      std::function<absl::Status(Any, std::vector<Element>, DoContext&)>
          custom_do) {
    custom_do_ = std::move(custom_do);
  }

 private:
  std::optional<absl::Status> do_error_;
  std::function<absl::Status(Any, std::vector<Element>, DoContext&)> custom_do_;
};

class BatchDoFnTest : public testing::Test {
 protected:
  TestBatchDoFn fn_;
  StrictMock<MockContext> context_;
  Counters counters_;
};

TEST_F(BatchDoFnTest, WriteAccumulatesData) {
  WriteRequest request;
  auto result = fn_.Write(request, "data_1", context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->committed_size_bytes(), 6);  // "data_1" is 6 bytes

  auto result2 = fn_.Write(request, "data_2", context_);
  ASSERT_THAT(result2, IsOk());
  EXPECT_EQ(result2->committed_size_bytes(), 6);
}

TEST_F(BatchDoFnTest, WritePreservesKeyAndBlobId) {
  WriteRequest request;
  Any key;
  key.set_type_url("test_type");
  key.set_value("test_key_value");
  *request.mutable_first_request_configuration() = key;

  // Set up an unencrypted blob header with a blob_id.
  auto* unencrypted =
      request.mutable_first_request_metadata()->mutable_unencrypted();
  unencrypted->set_blob_id("blob_123");

  auto result = fn_.Write(request, "payload", context_);
  ASSERT_THAT(result, IsOk());

  // Verify key and blob_id are preserved by checking the per-input emission.
  // First emission is the summary, second is the individual input.
  EXPECT_CALL(context_, EmitUnencrypted(_))
      .WillOnce(Return(true))  // summary emission
      .WillOnce([](Session::KV kv) {
        EXPECT_EQ(kv.data, "payload");
        EXPECT_EQ(kv.blob_id, "blob_123");
        EXPECT_EQ(kv.key.type_url(), "test_type");
        EXPECT_EQ(kv.key.value(), "test_key_value");
        return true;
      });
  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
}

TEST_F(BatchDoFnTest, CommitCallsDoWithAllInputs) {
  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "chunk_a", context_), IsOk());
  ASSERT_THAT(fn_.Write(request, "chunk_b", context_), IsOk());
  ASSERT_THAT(fn_.Write(request, "chunk_c", context_), IsOk());

  // Verify Do() received all 3 inputs by checking the emitted output.
  // 1 summary emission + 3 per-input emissions.
  EXPECT_CALL(context_, EmitUnencrypted(_))
      .WillOnce([](Session::KV kv) {
        // Data should be concatenation of all inputs.
        EXPECT_EQ(kv.data, "chunk_achunk_bchunk_c");
        // Input count should be 3.
        EXPECT_EQ(kv.key.value(), "3");
        return true;
      })
      .WillRepeatedly(Return(true));  // per-input emissions
  fcp::confidentialcompute::CommitRequest commit_request;
  auto commit_response = fn_.Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 3);
}

TEST_F(BatchDoFnTest, CommitForwardsConfig) {
  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "data", context_), IsOk());

  Any commit_config;
  commit_config.set_type_url("commit_type");
  commit_config.set_value("commit_value");
  fcp::confidentialcompute::CommitRequest commit_request;
  *commit_request.mutable_configuration() = commit_config;

  // Verify config was forwarded by checking blob_id in emitted output.
  // 1 summary emission + 1 per-input emission.
  EXPECT_CALL(context_, EmitUnencrypted(_))
      .WillOnce([](Session::KV kv) {
        EXPECT_EQ(kv.blob_id, "commit_type");
        return true;
      })
      .WillRepeatedly(Return(true));  // per-input emission
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
}

TEST_F(BatchDoFnTest, CommitWithNoWriteCallsDoWithEmptyList) {
  // Do() is still called — it's up to the subclass to decide if empty
  // input is an error.
  EXPECT_CALL(context_, EmitUnencrypted(_)).WillOnce([](Session::KV kv) {
    EXPECT_EQ(kv.data, "");
    EXPECT_EQ(kv.key.value(), "0");
    return true;
  });
  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
}

TEST_F(BatchDoFnTest, CommitReturnsDoError) {
  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "data", context_), IsOk());

  fn_.SetDoError(absl::InternalError("computation failed"));
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_.Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(BatchDoFnTest, FullLifecycleViaConfigure) {
  // Test through the Fn::Configure → Write → Commit → Finalize path.
  fcp::confidentialcompute::ConfigureRequest configure_request;
  auto configure_result = fn_.Configure(configure_request, context_);
  ASSERT_THAT(configure_result, IsOk());

  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "input_1", context_), IsOk());
  ASSERT_THAT(fn_.Write(request, "input_2", context_), IsOk());

  // 1 summary emission + 2 per-input emissions.
  EXPECT_CALL(context_, EmitUnencrypted(_))
      .WillOnce([](Session::KV kv) {
        EXPECT_EQ(kv.data, "input_1input_2");
        EXPECT_EQ(kv.key.value(), "2");
        return true;
      })
      .WillRepeatedly(Return(true));  // per-input emissions

  fcp::confidentialcompute::CommitRequest commit_request;
  auto commit_result = fn_.Commit(commit_request, context_);
  ASSERT_THAT(commit_result, IsOk());

  // Finalize is a no-op after Commit.
  fcp::confidentialcompute::FinalizeRequest finalize_request;
  fcp::confidentialcompute::BlobMetadata input_metadata;
  auto finalize_result =
      fn_.Finalize(finalize_request, input_metadata, context_);
  ASSERT_THAT(finalize_result, IsOk());
}

WriteRequest CreateWriteRequestWithMetadata(const std::string& blob_id,
                                            const std::string& key_id) {
  WriteRequest request;
  BlobHeader header;
  header.set_blob_id(blob_id);
  header.set_key_id(key_id);
  AssociatedMetadata assoc_metadata;
  assoc_metadata.add_metadata()->PackFrom(header);
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(assoc_metadata);
  return request;
}

TEST_F(BatchDoFnTest, WritePreservesAssociatedMetadataInInput) {
  ASSERT_THAT(fn_.Write(CreateWriteRequestWithMetadata("meta_blob_id_1",
                                                       "meta_key_id_1"),
                        "data_1", context_),
              IsOk());
  ASSERT_THAT(fn_.Write(CreateWriteRequestWithMetadata("meta_blob_id_2",
                                                       "meta_key_id_2"),
                        "data_2", context_),
              IsOk());

  bool metadata_verified = false;
  fn_.SetCustomDo([&](Any config, std::vector<Element> inputs,
                      DoContext& context) -> absl::Status {
    EXPECT_THAT(inputs, SizeIs(2));
    BlobHeader header_1;
    EXPECT_EQ(inputs[0].associated_metadata.metadata_size(), 1);
    EXPECT_TRUE(inputs[0].associated_metadata.metadata_size() > 0 &&
                inputs[0].associated_metadata.metadata(0).UnpackTo(&header_1));
    EXPECT_EQ(header_1.blob_id(), "meta_blob_id_1");
    EXPECT_EQ(header_1.key_id(), "meta_key_id_1");
    EXPECT_EQ(inputs[0].kv.data, "data_1");
    BlobHeader header_2;
    EXPECT_EQ(inputs[1].associated_metadata.metadata_size(), 1);
    EXPECT_TRUE(inputs[1].associated_metadata.metadata_size() > 0 &&
                inputs[1].associated_metadata.metadata(0).UnpackTo(&header_2));
    EXPECT_EQ(header_2.blob_id(), "meta_blob_id_2");
    EXPECT_EQ(header_2.key_id(), "meta_key_id_2");
    EXPECT_EQ(inputs[1].kv.data, "data_2");
    metadata_verified = true;
    return absl::OkStatus();
  });

  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
  EXPECT_TRUE(metadata_verified);
}

TEST_F(BatchDoFnTest, WritePreservesProtectedMetadataInInput) {
  BlobHeader protected_header;
  protected_header.set_blob_id("protected_blob_id");
  ProtectedMetadata protected_metadata;
  protected_metadata.add_metadata()->PackFrom(protected_header);
  ASSERT_THAT(fn_.Write(WriteRequest(), "data", protected_metadata, context_),
              IsOk());

  bool metadata_verified = false;
  fn_.SetCustomDo([&](Any config, std::vector<Element> inputs,
                      DoContext& context) -> absl::Status {
    EXPECT_THAT(inputs, SizeIs(1));
    BlobHeader unpacked;
    EXPECT_EQ(inputs[0].protected_metadata.metadata_size(), 1);
    EXPECT_TRUE(inputs[0].protected_metadata.metadata_size() > 0 &&
                inputs[0].protected_metadata.metadata(0).UnpackTo(&unpacked));
    EXPECT_EQ(unpacked.blob_id(), "protected_blob_id");
    EXPECT_EQ(inputs[0].associated_metadata.metadata_size(), 0);
    metadata_verified = true;
    return absl::OkStatus();
  });

  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
  EXPECT_TRUE(metadata_verified);
}

TEST_F(BatchDoFnTest, SetAssociatedMetadataFromInputPropagatesMetadata) {
  ASSERT_THAT(
      fn_.Write(CreateWriteRequestWithMetadata("meta_blob_id", "meta_key_id"),
                "data", context_),
      IsOk());

  Session::KV emitted_kv;
  AssociatedMetadata emitted_metadata;
  EXPECT_CALL(context_, EmitEncrypted(0, _, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv),
                      SaveAssociatedMetadataArg<2>(&emitted_metadata),
                      Return(true)));

  fn_.SetCustomDo([&](Any config, std::vector<Element> inputs,
                      DoContext& context) -> absl::Status {
    context.associated_metadata().Set(inputs[0].associated_metadata);
    EXPECT_TRUE(context.EmitEncrypted(0, std::move(inputs[0].kv)));
    return absl::OkStatus();
  });

  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
  EXPECT_EQ(emitted_kv.data, "data");
  BlobHeader emitted_header;
  ASSERT_EQ(emitted_metadata.metadata_size(), 1);
  ASSERT_TRUE(emitted_metadata.metadata(0).UnpackTo(&emitted_header));
  EXPECT_EQ(emitted_header.blob_id(), "meta_blob_id");
}

TEST_F(BatchDoFnTest, SetAssociatedMetadataReplacesPreviousInputMetadata) {
  ASSERT_THAT(
      fn_.Write(CreateWriteRequestWithMetadata("meta_blob_id", "meta_key_id"),
                "data_1", context_),
      IsOk());
  // The second input has no associated metadata.
  ASSERT_THAT(fn_.Write(WriteRequest(), "data_2", context_), IsOk());

  // The second output must not inherit the first input's metadata.
  AssociatedMetadata emitted_metadata;
  EXPECT_CALL(context_, EmitEncrypted(0, _, _))
      .WillOnce(
          DoAll(SaveAssociatedMetadataArg<2>(&emitted_metadata), Return(true)));
  Session::KV emitted_without_metadata;
  EXPECT_CALL(context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_without_metadata), Return(true)));

  fn_.SetCustomDo([&](Any config, std::vector<Element> inputs,
                      DoContext& context) -> absl::Status {
    for (Element& input : inputs) {
      context.associated_metadata().Set(std::move(input.associated_metadata));
      context.protected_metadata().Set(std::move(input.protected_metadata));
      EXPECT_TRUE(context.EmitEncrypted(0, std::move(input.kv)));
    }
    return absl::OkStatus();
  });

  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
  BlobHeader emitted_header;
  ASSERT_EQ(emitted_metadata.metadata_size(), 1);
  ASSERT_TRUE(emitted_metadata.metadata(0).UnpackTo(&emitted_header));
  EXPECT_EQ(emitted_header.blob_id(), "meta_blob_id");
  EXPECT_EQ(emitted_without_metadata.data, "data_2");
}

TEST_F(BatchDoFnTest, DoContextStartsWithEmptyMetadata) {
  ASSERT_THAT(
      fn_.Write(CreateWriteRequestWithMetadata("meta_blob_id", "meta_key_id"),
                "data", context_),
      IsOk());

  bool empty_metadata_verified = false;
  EXPECT_CALL(context_, EmitEncrypted(0, _)).WillOnce(Return(true));

  fn_.SetCustomDo([&](Any config, std::vector<Element> inputs,
                      DoContext& context) -> absl::Status {
    BlobHeader unpacked_header;
    EXPECT_FALSE(context.associated_metadata().Unpack(&unpacked_header));

    Session::KV output;
    output.data = "output_data";
    context.EmitEncrypted(0, std::move(output));
    empty_metadata_verified = true;
    return absl::OkStatus();
  });

  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
  EXPECT_TRUE(empty_metadata_verified);
}

}  // namespace
}  // namespace confidential_federated_compute::fns
