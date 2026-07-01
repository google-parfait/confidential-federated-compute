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

#include "containers/construct_user_session/row_gather.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "containers/construct_user_session/checkpoint.h"
#include "fcp/confidentialcompute/constants.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/in_memory_checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {
namespace {

using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::SizeIs;

const std::string kEventTime = absl::StrCat(
    "test_query/", fcp::confidential_compute::kEventTimeColumnName);

// Returns an absl::Time that is n hours after the Unix epoch.
absl::Time Hours(int n) { return absl::UnixEpoch() + absl::Hours(n); }

// Formats an absl::Time as an RFC3339 string for use as an event time.
// Pairs naturally with Hours(), e.g. EventTimeAt(Hours(5)).
std::string EventTimeAt(absl::Time t) {
  return absl::FormatTime(absl::RFC3339_full, t, absl::UTCTimeZone());
}

Checkpoint MakeCheckpoint(
    std::string privacy_id, std::vector<std::string> event_times,
    absl::flat_hash_map<std::string, Tensor> data_tensors) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor(std::move(privacy_id),
                           fcp::confidential_compute::kPrivacyIdColumnName));
  tensors.push_back(Tensor(std::move(event_times), kEventTime));
  for (auto& [name, tensor] : data_tensors) {
    tensors.push_back(std::move(tensor));
  }

  tensorflow_federated::aggregation::InMemoryCheckpointParser parser(
      std::move(tensors));
  absl::StatusOr<Checkpoint> cp = Checkpoint::Create(parser, "test_query");
  CHECK_OK(cp);
  return *std::move(cp);
}

// Verify that the result contains the expected ISO-8601 event time strings.
void VerifyEventTimes(const absl::flat_hash_map<std::string, Tensor>& result,
                      absl::string_view tensor_name,
                      const std::vector<std::string>& expected_strings) {
  ASSERT_TRUE(result.contains(tensor_name))
      << "Missing tensor: " << tensor_name;
  auto span = result.at(std::string(tensor_name)).AsSpan<absl::string_view>();
  std::vector<std::string> actual(span.begin(), span.end());
  EXPECT_THAT(actual, ElementsAreArray(expected_strings));
}

TEST(GatherSessionRowsTest, SingleCheckpointAllRows) {
  // Three events at Hours(5), Hours(10), Hours(15) — all inside the
  // window [Hours(0), Hours(24)).
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("colA", Tensor(std::vector<int32_t>{1, 2, 3}, "colA"));
  data_tensors.emplace("colB",
                       Tensor(std::vector<float>{1.5f, 2.5f, 3.5f}, "colB"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(MakeCheckpoint(
      "user1",
      {EventTimeAt(Hours(5)), EventTimeAt(Hours(10)), EventTimeAt(Hours(15))},
      std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
      {"colB", DataType::DT_FLOAT},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, SizeIs(3));
  ASSERT_TRUE(result.contains("colA"));
  EXPECT_THAT(result.at("colA").AsSpan<int32_t>(), ElementsAre(1, 2, 3));

  ASSERT_TRUE(result.contains("colB"));
  EXPECT_THAT(result.at("colB").AsSpan<float>(), ElementsAre(1.5f, 2.5f, 3.5f));

  VerifyEventTimes(
      result, kEventTime,
      {EventTimeAt(Hours(5)), EventTimeAt(Hours(10)), EventTimeAt(Hours(15))});
}

TEST(GatherSessionRowsTest, MultipleCheckpointsAllRows) {
  // Two checkpoints, all events inside the window.
  absl::flat_hash_map<std::string, Tensor> data_tensors1;
  data_tensors1.emplace("colA", Tensor(std::vector<int32_t>{10, 20}, "colA"));

  absl::flat_hash_map<std::string, Tensor> data_tensors2;
  data_tensors2.emplace("colA", Tensor(std::vector<int32_t>{30}, "colA"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(
      MakeCheckpoint("user1", {EventTimeAt(Hours(5)), EventTimeAt(Hours(10))},
                     std::move(data_tensors1)));
  checkpoints.push_back(MakeCheckpoint("user1", {EventTimeAt(Hours(15))},
                                       std::move(data_tensors2)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, SizeIs(2));
  ASSERT_TRUE(result.contains("colA"));
  EXPECT_THAT(result.at("colA").AsSpan<int32_t>(), ElementsAre(10, 20, 30));
  VerifyEventTimes(
      result, kEventTime,
      {EventTimeAt(Hours(5)), EventTimeAt(Hours(10)), EventTimeAt(Hours(15))});
}

TEST(GatherSessionRowsTest, TimeWindowFiltersRows) {
  // Window: [Hours(10), Hours(20)).
  // Two events inside the window (Hours(12) and Hours(15)) and two outside
  // (Hours(5) and Hours(25)).
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("colA",
                       Tensor(std::vector<int32_t>{10, 20, 30, 40}, "colA"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(
      MakeCheckpoint("user1",
                     {EventTimeAt(Hours(12)),   // in window
                      EventTimeAt(Hours(5)),    // before window
                      EventTimeAt(Hours(15)),   // in window
                      EventTimeAt(Hours(25))},  // after window
                     std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(10), Hours(20),
                                  column_types);

  EXPECT_THAT(result, SizeIs(2));
  ASSERT_TRUE(result.contains("colA"));
  // Only rows 0 and 2 survive (values 10 and 30).
  EXPECT_THAT(result.at("colA").AsSpan<int32_t>(), ElementsAre(10, 30));
}

TEST(GatherSessionRowsTest, AllRowsFilteredReturnsEmpty) {
  // Window: [Hours(10), Hours(20)). Both events are before the window.
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("colA", Tensor(std::vector<int32_t>{1, 2}, "colA"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(
      MakeCheckpoint("user1", {EventTimeAt(Hours(2)), EventTimeAt(Hours(5))},
                     std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(10), Hours(20),
                                  column_types);

  EXPECT_THAT(result, IsEmpty());
}

TEST(GatherSessionRowsTest, HeterogeneousTensorSets) {
  absl::flat_hash_map<std::string, Tensor> data_tensors1;
  data_tensors1.emplace("foo", Tensor(std::vector<int32_t>{1}, "foo"));
  data_tensors1.emplace("bar", Tensor(std::vector<int32_t>{10}, "bar"));

  absl::flat_hash_map<std::string, Tensor> data_tensors2;
  data_tensors2.emplace("foo", Tensor(std::vector<int32_t>{2}, "foo"));
  data_tensors2.emplace("baz", Tensor(std::vector<int32_t>{20}, "baz"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(MakeCheckpoint("user1", {EventTimeAt(Hours(5))},
                                       std::move(data_tensors1)));
  checkpoints.push_back(MakeCheckpoint("user1", {EventTimeAt(Hours(10))},
                                       std::move(data_tensors2)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"foo", DataType::DT_INT32},
      {"bar", DataType::DT_INT32},
      {"baz", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, SizeIs(4));
  EXPECT_THAT(result.at("foo").AsSpan<int32_t>(), ElementsAre(1, 2));
  EXPECT_THAT(result.at("bar").AsSpan<int32_t>(), ElementsAre(10));
  EXPECT_THAT(result.at("baz").AsSpan<int32_t>(), ElementsAre(20));
}

TEST(GatherSessionRowsTest, AbsentTensorSkipsColumn) {
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("foo", Tensor(std::vector<int32_t>{1}, "foo"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(MakeCheckpoint("user1", {EventTimeAt(Hours(5))},
                                       std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"foo", DataType::DT_INT32},
      {"absent", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, SizeIs(2));
  ASSERT_TRUE(result.contains("foo"));
  EXPECT_FALSE(result.contains("absent"));
}

TEST(GatherSessionRowsTest, EmptyCheckpoints) {
  std::vector<Checkpoint> checkpoints;  // empty

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, IsEmpty());
}

TEST(GatherSessionRowsTest, MixedTypes) {
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("colInt32", Tensor(std::vector<int32_t>{1}, "colInt32"));
  data_tensors.emplace("colInt64", Tensor(std::vector<int64_t>{2}, "colInt64"));
  data_tensors.emplace("colUint64",
                       Tensor(std::vector<uint64_t>{3}, "colUint64"));
  data_tensors.emplace("colFloat",
                       Tensor(std::vector<float>{4.5f}, "colFloat"));
  data_tensors.emplace("colDouble",
                       Tensor(std::vector<double>{5.5}, "colDouble"));
  data_tensors.emplace("colString",
                       Tensor(std::vector<std::string>{"mixed"}, "colString"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(MakeCheckpoint("user1", {EventTimeAt(Hours(5))},
                                       std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},  {"colInt32", DataType::DT_INT32},
      {"colInt64", DataType::DT_INT64},   {"colUint64", DataType::DT_UINT64},
      {"colFloat", DataType::DT_FLOAT},   {"colDouble", DataType::DT_DOUBLE},
      {"colString", DataType::DT_STRING},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  EXPECT_THAT(result, SizeIs(7));
  EXPECT_THAT(result.at("colInt32").AsSpan<int32_t>(), ElementsAre(1));
  EXPECT_THAT(result.at("colInt64").AsSpan<int64_t>(), ElementsAre(2));
  EXPECT_THAT(result.at("colUint64").AsSpan<uint64_t>(), ElementsAre(3));
  EXPECT_THAT(result.at("colFloat").AsSpan<float>(), ElementsAre(4.5f));
  EXPECT_THAT(result.at("colDouble").AsSpan<double>(), ElementsAre(5.5));
  EXPECT_THAT(result.at("colString").ToStringVector(), ElementsAre("mixed"));
}

TEST(GatherSessionRowsTest, MalformedTimestampsExcluded) {
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.emplace("colA",
                       Tensor(std::vector<int32_t>{10, 20, 30}, "colA"));

  std::vector<Checkpoint> checkpoints;
  checkpoints.push_back(MakeCheckpoint("user1",
                                       {"not-a-timestamp",      // malformed
                                        EventTimeAt(Hours(5)),  // valid
                                        "also-invalid"},        // malformed
                                       std::move(data_tensors)));

  absl::flat_hash_map<std::string, DataType> column_types = {
      {kEventTime, DataType::DT_STRING},
      {"colA", DataType::DT_INT32},
  };

  auto result = GatherSessionRows(checkpoints, kEventTime, Hours(0), Hours(24),
                                  column_types);

  // Only the valid timestamp's row survives.
  EXPECT_THAT(result, SizeIs(2));
  ASSERT_TRUE(result.contains("colA"));
  EXPECT_THAT(result.at("colA").AsSpan<int32_t>(), ElementsAre(20));
}

}  // namespace
}  // namespace confidential_federated_compute::construct_user_session
