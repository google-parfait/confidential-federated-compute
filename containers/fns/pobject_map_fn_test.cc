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

#include "containers/fns/pobject_map_fn.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::StrictMock;

// Mock Context for verifying emissions.
class MockContext : public Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
  MOCK_METHOD(bool, EmitUnencrypted, (Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV), (override));
  MOCK_METHOD(bool, EmitReleasable,
              (int, Session::KV, std::optional<absl::string_view>,
               absl::string_view, std::string&),
              (override));
  MOCK_METHOD(Counters&, GetCounters, (), (override));
};

// A concrete subclass for testing. Records the inputs received by Map()
// and can be configured to emit specific outputs or return errors.
class TestPObjectMapFn : public PObjectMapFn {
 public:
  absl::Status Map(Any config, std::vector<Session::KV> accumulated_inputs,
                   Context& context) override {
    if (map_error_.has_value()) {
      return *map_error_;
    }

    // Emit a summary: concatenate all input data, keys, and blob_ids.
    std::string combined;
    for (const auto& kv : accumulated_inputs) {
      combined += kv.data;
    }
    Session::KV output;
    output.data = std::move(combined);
    // Encode config into the output so tests can verify it was forwarded.
    output.blob_id = config.type_url();
    // Encode input count into the key.
    Any count_key;
    count_key.set_value(std::to_string(accumulated_inputs.size()));
    output.key = std::move(count_key);
    context.EmitUnencrypted(std::move(output));

    // Also emit each input individually so tests can verify key/blob_id.
    for (auto& kv : accumulated_inputs) {
      context.EmitUnencrypted(std::move(kv));
    }
    return absl::OkStatus();
  }

  // Configure Map() to return an error.
  void SetMapError(absl::Status error) { map_error_ = std::move(error); }

 private:
  std::optional<absl::Status> map_error_;
};

class PObjectMapFnTest : public testing::Test {
 protected:
  TestPObjectMapFn fn_;
  StrictMock<MockContext> context_;
  Counters counters_;
};

TEST_F(PObjectMapFnTest, WriteAccumulatesData) {
  WriteRequest request;
  auto result = fn_.Write(request, "data_1", context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->committed_size_bytes(), 6);  // "data_1" is 6 bytes

  auto result2 = fn_.Write(request, "data_2", context_);
  ASSERT_THAT(result2, IsOk());
  EXPECT_EQ(result2->committed_size_bytes(), 6);
}

TEST_F(PObjectMapFnTest, WritePreservesKeyAndBlobId) {
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

TEST_F(PObjectMapFnTest, CommitCallsMapWithAllInputs) {
  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "chunk_a", context_), IsOk());
  ASSERT_THAT(fn_.Write(request, "chunk_b", context_), IsOk());
  ASSERT_THAT(fn_.Write(request, "chunk_c", context_), IsOk());

  // Verify Map() received all 3 inputs by checking the emitted output.
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

TEST_F(PObjectMapFnTest, CommitForwardsConfig) {
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

TEST_F(PObjectMapFnTest, CommitWithNoWriteCallsMapWithEmptyList) {
  // Map() is still called — it's up to the subclass to decide if empty
  // input is an error.
  EXPECT_CALL(context_, EmitUnencrypted(_)).WillOnce([](Session::KV kv) {
    EXPECT_EQ(kv.data, "");
    EXPECT_EQ(kv.key.value(), "0");
    return true;
  });
  fcp::confidentialcompute::CommitRequest commit_request;
  ASSERT_THAT(fn_.Commit(commit_request, context_), IsOk());
}

TEST_F(PObjectMapFnTest, CommitReturnsMapError) {
  WriteRequest request;
  ASSERT_THAT(fn_.Write(request, "data", context_), IsOk());

  fn_.SetMapError(absl::InternalError("computation failed"));
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_.Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(PObjectMapFnTest, FullLifecycleViaConfigure) {
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

}  // namespace
}  // namespace confidential_federated_compute::fns
