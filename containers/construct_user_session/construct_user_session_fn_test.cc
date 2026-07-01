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

#include "containers/construct_user_session/construct_user_session_fn.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/testing/mocks.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/util/time_util.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::ConstructUserSessionInitConfig;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::google::protobuf::Timestamp;
using ::google::protobuf::util::TimeUtil;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Test;

constexpr absl::string_view kQueryName = "test_query";

// Returns an absl::Time that is n hours after the Unix epoch.
absl::Time Hours(int n) { return absl::UnixEpoch() + absl::Hours(n); }

// Formats an absl::Time as an RFC3339 string for use as an event time in
// BuildCheckpoint. Pairs naturally with Hours(), e.g. EventTimeAt(Hours(12)).
std::string EventTimeAt(absl::Time t) {
  return absl::FormatTime(absl::RFC3339_full, t, absl::UTCTimeZone());
}

// Builds a serialized FederatedCompute checkpoint with the given privacy ID,
// event times, and optional extra column tensors.
std::string BuildCheckpoint(
    absl::string_view privacy_id, const std::vector<std::string>& event_times,
    absl::string_view on_device_query_name,
    const std::map<std::string, std::vector<int32_t>>& extra_int_columns = {}) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      builder = builder_factory.Create();

  // Privacy ID scalar tensor.
  Tensor pid_tensor(std::string{privacy_id}, std::string{kPrivacyIdColumnName});
  CHECK_OK(builder->Add(kPrivacyIdColumnName, std::move(pid_tensor)));

  // Event time tensor.
  std::string event_time_name =
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName);
  Tensor event_tensor{std::vector<std::string>(event_times), event_time_name};
  CHECK_OK(builder->Add(event_time_name, std::move(event_tensor)));

  // Extra int32 columns.
  for (const auto& [col_name, values] : extra_int_columns) {
    std::string full_name = absl::StrCat(on_device_query_name, "/", col_name);
    Tensor col_tensor{std::vector<int32_t>(values), full_name};
    CHECK_OK(builder->Add(full_name, std::move(col_tensor)));
  }

  absl::StatusOr<absl::Cord> result = builder->Build();
  CHECK_OK(result);
  return std::string(result->Flatten());
}

// Helper to create a ConstructUserSessionInitConfig packed in an Any.
Any MakeValidConfig(
    absl::Time window_start = Hours(0), absl::Time window_end = Hours(24),
    const std::string& on_device_query_name = std::string(kQueryName)) {
  ConstructUserSessionInitConfig config;
  *config.mutable_session_window_start() =
      TimeUtil::NanosecondsToTimestamp(absl::ToUnixNanos(window_start));
  *config.mutable_session_window_end() =
      TimeUtil::NanosecondsToTimestamp(absl::ToUnixNanos(window_end));
  config.set_on_device_query_name(on_device_query_name);
  Any any;
  any.PackFrom(config);
  return any;
}

// Helper to call Write on the Fn with a given checkpoint string and blob_id.
absl::StatusOr<WriteFinishedResponse> DoWrite(
    fns::Fn& fn, MockContext& context, const std::string& checkpoint_data,
    const std::string& blob_id = "blob_1") {
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id(blob_id);
  return fn.Write(write_request, checkpoint_data, context);
}

// Parses a serialized FedCompute checkpoint and returns tensors by name.
absl::flat_hash_map<std::string, Tensor> ParseCheckpoint(
    const std::string& data) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(data));
  CHECK_OK(parser);
  auto tensors = (*parser)->LoadAllTensors();
  CHECK_OK(tensors);
  return *std::move(tensors);
}

class ConstructUserSessionFnFactoryTest : public Test {};

TEST_F(ConstructUserSessionFnFactoryTest, FailsWithUnpackableConfiguration) {
  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  Any(), Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot be unpacked")));
}

TEST_F(ConstructUserSessionFnFactoryTest, FailsWithMissingWindowStart) {
  ConstructUserSessionInitConfig config;
  Timestamp ts_end;
  CHECK(TimeUtil::FromString("2026-01-02T00:00:00Z", &ts_end));
  *config.mutable_session_window_end() = ts_end;
  config.set_on_device_query_name("test_query");
  Any any;
  any.PackFrom(config);

  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  any, Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("session_window_start")));
}

TEST_F(ConstructUserSessionFnFactoryTest, FailsWithMissingWindowEnd) {
  ConstructUserSessionInitConfig config;
  Timestamp ts_start;
  CHECK(TimeUtil::FromString("2026-01-01T00:00:00Z", &ts_start));
  *config.mutable_session_window_start() = ts_start;
  config.set_on_device_query_name("test_query");
  Any any;
  any.PackFrom(config);

  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  any, Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("session_window_end")));
}

TEST_F(ConstructUserSessionFnFactoryTest, FailsWhenWindowStartEqualsWindowEnd) {
  Any config = MakeValidConfig(Hours(5), Hours(5));
  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  config, Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("strictly before")));
}

TEST_F(ConstructUserSessionFnFactoryTest, FailsWhenWindowStartAfterWindowEnd) {
  Any config = MakeValidConfig(Hours(10), Hours(5));
  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  config, Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("strictly before")));
}

TEST_F(ConstructUserSessionFnFactoryTest, FailsWithEmptyOnDeviceQueryName) {
  ConstructUserSessionInitConfig config;
  Timestamp ts_start, ts_end;
  CHECK(TimeUtil::FromString("2026-01-01T00:00:00Z", &ts_start));
  CHECK(TimeUtil::FromString("2026-01-02T00:00:00Z", &ts_end));
  *config.mutable_session_window_start() = ts_start;
  *config.mutable_session_window_end() = ts_end;
  // on_device_query_name left empty.
  Any any;
  any.PackFrom(config);

  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  any, Any(), WriteConfigurationMap{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("on_device_query_name")));
}

TEST_F(ConstructUserSessionFnFactoryTest, SuccessfulCreation) {
  Any config = MakeValidConfig();
  EXPECT_THAT(CreateConstructUserSessionFnFactoryProvider(
                  config, Any(), WriteConfigurationMap{}),
              IsOk());
}

TEST_F(ConstructUserSessionFnFactoryTest, CreateFnProducesInstances) {
  Any config = MakeValidConfig();
  auto factory = CreateConstructUserSessionFnFactoryProvider(
      config, Any(), WriteConfigurationMap{});
  ASSERT_THAT(factory, IsOk());
  EXPECT_THAT((*factory)->CreateFn(), IsOk());
  EXPECT_THAT((*factory)->CreateFn(), IsOk());
}

class ConstructUserSessionFnTest : public Test {
 protected:
  void SetUp() override {
    Any config = MakeValidConfig();
    auto factory = CreateConstructUserSessionFnFactoryProvider(
        config, Any(), WriteConfigurationMap{});
    CHECK_OK(factory);
    auto fn = (*factory)->CreateFn();
    CHECK_OK(fn);
    fn_ = *std::move(fn);
  }

  std::unique_ptr<fns::Fn> fn_;
  StrictMock<MockContext> context_;
};

TEST_F(ConstructUserSessionFnTest, SingleBlobWithInWindowRowsPassesThrough) {
  std::string checkpoint = BuildCheckpoint("user_1", {EventTimeAt(Hours(12))},
                                           kQueryName, {{"score", {42}}});

  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint), IsOk());

  // Commit should emit exactly one checkpoint that is a pass-through of the
  // input (column order in the serialized bytes may differ).
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_kv](int, Session::KV kv) {
        emitted_kv = std::move(kv);
        return true;
      });

  ASSERT_THAT(fn_->Commit(CommitRequest(), context_), IsOk());

  auto tensors = ParseCheckpoint(emitted_kv.data);
  EXPECT_EQ(tensors.at(std::string(kPrivacyIdColumnName))
                .AsScalar<absl::string_view>(),
            "user_1");
  EXPECT_THAT(tensors.at(absl::StrCat(kQueryName, "/score")).AsSpan<int32_t>(),
              ElementsAre(42));
}

TEST_F(ConstructUserSessionFnTest, DuplicateBlobIdIsSkipped) {
  std::string checkpoint =
      BuildCheckpoint("user_1", {EventTimeAt(Hours(12))}, kQueryName);

  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint, "blob_dup"), IsOk());
  // Write the same blob_id again — PObjectMapFn stashes it; dedup happens
  // in Map() during Commit.
  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint, "blob_dup"), IsOk());

  // Only one checkpoint should be emitted. GetCounters() is called during
  // Map() (invoked by Commit) to increment the duplicate_blob_count.
  EXPECT_CALL(context_, GetCounters())
      .WillRepeatedly(testing::ReturnRef(context_.counters_));
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_kv](int, Session::KV kv) {
        emitted_kv = std::move(kv);
        return true;
      });
  auto commit_result = fn_->Commit(CommitRequest(), context_);
  ASSERT_THAT(commit_result, IsOk());

  // The emitted checkpoint must contain exactly 1 row (the first write).
  auto tensors = ParseCheckpoint(emitted_kv.data);
  EXPECT_EQ(tensors.at(absl::StrCat(kQueryName, "/", kEventTimeColumnName))
                .AsSpan<absl::string_view>()
                .size(),
            1);

  // Verify the duplicate_blob_count counter was incremented.
  EXPECT_EQ(context_.counters_["duplicate_blob_count"], 1);
}

TEST_F(ConstructUserSessionFnTest, DtypeMismatchSkipsCheckpoint) {
  // user_1's first checkpoint: test_query/col with DT_INT32.
  std::string int32_checkpoint = BuildCheckpoint(
      "user_1", {EventTimeAt(Hours(12))}, kQueryName, {{"col", {10}}});

  // user_1's second checkpoint: test_query/col with DT_STRING (different
  // dtype).
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  Tensor pid("user_1", std::string(kPrivacyIdColumnName));
  CHECK_OK(builder->Add(kPrivacyIdColumnName, std::move(pid)));

  std::string et_name = absl::StrCat(kQueryName, "/", kEventTimeColumnName);
  Tensor et({EventTimeAt(Hours(13))}, et_name);
  CHECK_OK(builder->Add(et_name, std::move(et)));

  std::string col_name = absl::StrCat(kQueryName, "/col");
  Tensor col({"string_val"}, col_name);
  CHECK_OK(builder->Add(col_name, std::move(col)));

  auto cord = builder->Build();
  CHECK_OK(cord);
  std::string string_checkpoint(cord->Flatten());

  ASSERT_THAT(DoWrite(*fn_, context_, int32_checkpoint, "blob_1"), IsOk());
  ASSERT_THAT(DoWrite(*fn_, context_, string_checkpoint, "blob_2"), IsOk());

  EXPECT_CALL(context_, GetCounters())
      .WillOnce(testing::ReturnRef(context_.counters_));
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_kv](int, Session::KV kv) {
        emitted_kv = std::move(kv);
        return true;
      });
  auto commit_result = fn_->Commit(CommitRequest(), context_);
  ASSERT_THAT(commit_result, IsOk());

  // We expect only the first checkpoint's row in the output.
  auto tensors = ParseCheckpoint(emitted_kv.data);
  EXPECT_EQ(tensors.at(absl::StrCat(kQueryName, "/", kEventTimeColumnName))
                .AsSpan<absl::string_view>()
                .size(),
            1);
  EXPECT_EQ(context_.counters_["checkpoint_dtype_mismatch_count"], 1);
}

TEST_F(ConstructUserSessionFnTest, AllRowsFilteredByTimeWindowDiscardsBlob) {
  // Event time is outside the [Hours(0), Hours(24)) window.
  std::string checkpoint = BuildCheckpoint("user_1", {EventTimeAt(Hours(48))},
                                           kQueryName, {{"score", {99}}});

  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint), IsOk());

  // Nothing should be emitted since all rows were filtered.
  auto commit_result = fn_->Commit(CommitRequest(), context_);
  ASSERT_THAT(commit_result, IsOk());
}

// Matches a Session::KV whose checkpoint has the given privacy ID and whose
// score column (kQueryName/score) satisfies the given int32 matcher.
MATCHER_P2(HasCheckpointFor, privacy_id, score_matcher,
           absl::StrCat("checkpoint for '", privacy_id, "' with scores ",
                        testing::PrintToString(score_matcher))) {
  auto tensors = ParseCheckpoint(arg.data);
  if (tensors.at(std::string(kPrivacyIdColumnName))
          .template AsScalar<absl::string_view>() != privacy_id) {
    return false;
  }
  std::string score_col = absl::StrCat(kQueryName, "/score");
  auto span = tensors.at(score_col).template AsSpan<int32_t>();
  return testing::ExplainMatchResult(
      score_matcher, std::vector<int32_t>(span.begin(), span.end()),
      result_listener);
}

TEST_F(ConstructUserSessionFnTest, MultipleBlobsSamePrivacyIdMerged) {
  // Two checkpoints for user_A with a user_B checkpoint written in between.
  // This tests that non-consecutive checkpoints for the same privacy ID are
  // still grouped and merged correctly.
  std::string checkpoint_a1 = BuildCheckpoint(
      "user_A", {EventTimeAt(Hours(10))}, kQueryName, {{"score", {10}}});
  std::string checkpoint_b = BuildCheckpoint("user_B", {EventTimeAt(Hours(11))},
                                             kQueryName, {{"score", {99}}});
  std::string checkpoint_a2 = BuildCheckpoint(
      "user_A", {EventTimeAt(Hours(14))}, kQueryName, {{"score", {20}}});

  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint_a1, "blob_a1"), IsOk());
  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint_b, "blob_b"), IsOk());
  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint_a2, "blob_a2"), IsOk());

  // Should emit two checkpoints total: one merged checkpoint for user_A and
  // one for user_B.
  std::vector<Session::KV> emitted_kvs;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .Times(2)
      .WillRepeatedly([&emitted_kvs](int, Session::KV kv) {
        emitted_kvs.push_back(std::move(kv));
        return true;
      });

  auto commit_result = fn_->Commit(CommitRequest(), context_);
  ASSERT_THAT(commit_result, IsOk());
  ASSERT_EQ(emitted_kvs.size(), 2);

  // user_A's two non-consecutive blobs should have been merged into one
  // checkpoint with both rows.
  EXPECT_THAT(emitted_kvs,
              Contains(HasCheckpointFor(
                  "user_A", testing::UnorderedElementsAre(10, 20))));

  // user_B's single blob should be present unchanged.
  EXPECT_THAT(emitted_kvs,
              Contains(HasCheckpointFor("user_B", ElementsAre(99))));
}

TEST_F(ConstructUserSessionFnTest, StateIsClearedAfterCommit) {
  std::string checkpoint =
      BuildCheckpoint("user_1", {EventTimeAt(Hours(12))}, kQueryName);

  // First round.
  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint, "blob_1"), IsOk());
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _)).WillOnce(Return(true));
  ASSERT_THAT(fn_->Commit(CommitRequest(), context_), IsOk());

  // Second round — same blob_id should be accepted (not treated as duplicate).
  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint, "blob_1"), IsOk());
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _)).WillOnce(Return(true));
  ASSERT_THAT(fn_->Commit(CommitRequest(), context_), IsOk());
}

TEST_F(ConstructUserSessionFnTest, RowsOutsideTimeWindowAreFilteredOut) {
  // Two event times: one inside [Hours(0), Hours(24)), one outside.
  std::string checkpoint = BuildCheckpoint(
      "user_1", {EventTimeAt(Hours(12)), EventTimeAt(Hours(48))}, kQueryName,
      {{"score", {10, 20}}});

  ASSERT_THAT(DoWrite(*fn_, context_, checkpoint), IsOk());

  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_kv](int, Session::KV kv) {
        emitted_kv = std::move(kv);
        return true;
      });

  ASSERT_THAT(fn_->Commit(CommitRequest(), context_), IsOk());

  auto tensors = ParseCheckpoint(emitted_kv.data);
  std::string score_name = absl::StrCat(kQueryName, "/score");
  ASSERT_TRUE(tensors.contains(score_name));
  // Only the row with event time inside the window should survive.
  EXPECT_THAT(tensors.at(score_name).AsSpan<int32_t>(), ElementsAre(10));
}

TEST_F(ConstructUserSessionFnTest, EmptyCommitEmitsNothing) {
  // Commit without any writes.
  auto commit_result = fn_->Commit(CommitRequest(), context_);
  ASSERT_THAT(commit_result, IsOk());
  // No EmitEncrypted calls expected (StrictMock would catch any).
}

}  // namespace
}  // namespace confidential_federated_compute::construct_user_session
