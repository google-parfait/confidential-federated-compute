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

#include "containers/construct_user_session/ingestion.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "fcp/confidentialcompute/constants.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/in_memory_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace confidential_federated_compute::construct_user_session {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::InMemoryCheckpointParser;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;

absl::Time Hours(int hours) { return absl::Time() + absl::Hours(hours); }

auto RowLocationIs(int group_key, int input_index, int row_index) {
  return FieldsAre(group_key, input_index, row_index);
}

TEST(DeserializeCheckpointTest, ValidCheckpoint) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(
      Tensor({"2026-01-01T10:00:00+00:00", "2026-01-01T10:01:00+00:00"},
             "test_query/confidential_compute_event_time"));
  tensors.push_back(Tensor({"valA1", "valA2"}, "test_query/colA"));
  tensors.push_back(Tensor({"valB1", "valB2"}, "colB"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id, Eq("user123"));
  EXPECT_THAT(
      result->event_times.ToStringVector(),
      ElementsAre("2026-01-01T10:00:00+00:00", "2026-01-01T10:01:00+00:00"));
  EXPECT_THAT(result->data_tensors, SizeIs(2));

  auto col_a_it = result->data_tensors.find("test_query/colA");
  ASSERT_NE(col_a_it, result->data_tensors.end());
  EXPECT_THAT(col_a_it->second.ToStringVector(), ElementsAre("valA1", "valA2"));

  auto col_b_it = result->data_tensors.find("colB");
  ASSERT_NE(col_b_it, result->data_tensors.end());
  EXPECT_THAT(col_b_it->second.ToStringVector(), ElementsAre("valB1", "valB2"));
}

TEST(DeserializeCheckpointTest, MissingPrivacyIdFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_privacy_id")));
}

TEST(DeserializeCheckpointTest, MissingEventTimeFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_event_time")));
}

TEST(DeserializeCheckpointTest, MultidimensionalDataTensorFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  auto data_tensor = Tensor::Create(
      DataType::DT_STRING, TensorShape({1, 1}),
      CreateTestData<absl::string_view>({"valA1"}), "test_query/colA");
  CHECK_OK(data_tensor);
  tensors.push_back(*std::move(data_tensor));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Data tensor `test_query/colA` must "
                                         "have one dimension.")));
}

TEST(DeserializeCheckpointTest, MismatchedRowCountsFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(
      Tensor({"2026-01-01T10:00:00+00:00", "2026-01-01T10:01:00+00:00"},
             "test_query/confidential_compute_event_time"));
  tensors.push_back(
      Tensor({"valA1"}, "test_query/colA"));  // 1 row instead of 2.

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Data tensor `test_query/colA` has 1 rows, "
                                 "expected 2 matching event time tensor.")));
}

// Verifies that a checkpoint with no data tensors (only system tensors)
// deserializes successfully with an empty data_tensors map.
TEST(DeserializeCheckpointTest, NoDataTensorsSucceeds) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user789", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-03-01T08:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id, Eq("user789"));
  EXPECT_EQ(result->event_times.num_elements(), 1);
  EXPECT_TRUE(result->data_tensors.empty());
}

// Verifies that timestamps are stored as-is in the raw strings
TEST(DeserializeCheckpointTest, EventTimeIsPreserved) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor(
      {"2026-01-01T10:00:00Z", "not-a-timestamp", "2026-01-01T10:02:00+14:00"},
      "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->event_times.ToStringVector(),
              ElementsAre("2026-01-01T10:00:00Z", "not-a-timestamp",
                          "2026-01-01T10:02:00+14:00"));
}

TEST(FilterForSessionWindowTest, AllEventsPassFilter) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times({"1970-01-01T06:00:00+00:00", "1970-01-01T12:00:00+00:00",
                      "1970-01-01T18:00:00+00:00"},
                     "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/42,
                             /*input_index=*/5, window_start, window_end);

  // One RowLocation is created for each event time.
  EXPECT_THAT(result,
              ElementsAre(RowLocationIs(42, 5, 0), RowLocationIs(42, 5, 1),
                          RowLocationIs(42, 5, 2)));
}

TEST(FilterForSessionWindowTest, AllEventsRejected) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times(
      {// Before window.
       "1969-12-31T23:00:00+00:00",
       // After window.
       "1970-01-02T01:00:00+00:00"},
      "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/1,
                             /*input_index=*/0, window_start, window_end);

  EXPECT_TRUE(result.empty());
}

TEST(FilterForSessionWindowTest, PartialPassFilter) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times({"1969-12-31T23:00:00+00:00",   // row 0: before window
                      "1970-01-01T12:00:00+00:00",   // row 1: in window
                      "1970-01-02T01:00:00+00:00"},  // row 2: after window
                     "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/99,
                             /*input_index=*/2, window_start, window_end);

  // Only row 1 survives.
  EXPECT_THAT(result, ElementsAre(RowLocationIs(99, 2, 1)));
}

TEST(FilterForSessionWindowTest, EmptyEventTimes) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times(std::vector<std::string>{}, "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/0,
                             /*input_index=*/0, window_start, window_end);

  EXPECT_TRUE(result.empty());
}

// Verifies that malformed timestamp strings are excluded by the filter.
TEST(FilterForSessionWindowTest, MalformedTimestampsAreExcluded) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times({"not-a-timestamp",  // row 0: malformed (excluded)
                      "1970-01-01T06:00:00+00:00",  // row 1: in window
                      "also-not-valid"},  // row 2: malformed (excluded)
                     "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/7,
                             /*input_index=*/3, window_start, window_end);

  EXPECT_THAT(result, ElementsAre(RowLocationIs(7, 3, 1)));
}

// Verifies that window_start is inclusive: an event exactly at window_start
// survives.
TEST(FilterForSessionWindowTest, WindowStartIsInclusive) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times({"1970-01-01T00:00:00+00:00"}, "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/100,
                             /*input_index=*/0, window_start, window_end);

  EXPECT_THAT(result, ElementsAre(RowLocationIs(100, 0, 0)));
}

// Verifies that window_end is exclusive: an event exactly at window_end is
// rejected.
TEST(FilterForSessionWindowTest, WindowEndIsExclusive) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(24);
  Tensor event_times({"1970-01-02T00:00:00+00:00"}, "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/100,
                             /*input_index=*/0, window_start, window_end);

  EXPECT_TRUE(result.empty());
}

// Verifies that FilterForSessionWindow correctly handles timestamps with
// different timezone offsets.
TEST(FilterForSessionWindowTest, DifferentOffsetsResolveToSameInstant) {
  absl::Time window_start = Hours(0);
  absl::Time window_end = Hours(1);

  Tensor event_times(
      {"1970-01-01T05:30:00+05:30",   // IST = 00:00 UTC (in window)
       "1970-01-01T00:00:00+00:00",   // UTC = 00:00 UTC (in window)
       "1970-01-01T12:00:00-08:00"},  // PST = 20:00 UTC (outside window)
      "event_time");

  std::vector<RowLocation> result =
      FilterForSessionWindow(event_times, /*group_key=*/1,
                             /*input_index=*/0, window_start, window_end);

  EXPECT_THAT(result,
              ElementsAre(RowLocationIs(1, 0, 0), RowLocationIs(1, 0, 1)));
}

}  // namespace
}  // namespace confidential_federated_compute::construct_user_session
