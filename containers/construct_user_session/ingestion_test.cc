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
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

// Parses an RFC3339 string to absl::Time for use in test assertions.
absl::Time ParseRfc3339(absl::string_view s) {
  absl::Time t;
  std::string err;
  CHECK(absl::ParseTime(absl::RFC3339_full, s, &t, &err)) << err;
  return t;
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
  auto result = DeserializeCheckpoint(&parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id, Eq("user123"));
  EXPECT_THAT(result->event_times,
              ElementsAre(ParseRfc3339("2026-01-01T10:00:00+00:00"),
                          ParseRfc3339("2026-01-01T10:01:00+00:00")));
  EXPECT_THAT(result->data_tensors.size(), Eq(2));

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
  auto result = DeserializeCheckpoint(&parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_privacy_id")));
}

TEST(DeserializeCheckpointTest, NullCheckpointFails) {
  EXPECT_THAT(DeserializeCheckpoint(nullptr, "test_query"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("checkpoint must not be null")));
}

TEST(DeserializeCheckpointTest, MissingEventTimeFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(&parser, "test_query");
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
  auto result = DeserializeCheckpoint(&parser, "test_query");
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
  auto result = DeserializeCheckpoint(&parser, "test_query");
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
  auto result = DeserializeCheckpoint(&parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id, Eq("user789"));
  EXPECT_THAT(result->event_times.size(), Eq(1));
  EXPECT_TRUE(result->data_tensors.empty());
}

// Verifies that malformed timestamps are stored as absl::InfinitePast()
// sentinels rather than causing a hard error, and valid timestamps in the same
// tensor are parsed correctly.
TEST(DeserializeCheckpointTest, MalformedEventTimeBecomesInfinitePast) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00", "not-a-timestamp",
                            "2026-01-01T10:02:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(&parser, "test_query");
  ASSERT_THAT(result, IsOk());

  ASSERT_THAT(result->event_times.size(), Eq(3));
  EXPECT_THAT(result->event_times[0],
              Eq(ParseRfc3339("2026-01-01T10:00:00+00:00")));
  EXPECT_THAT(result->event_times[1], Eq(absl::InfinitePast()));
  EXPECT_THAT(result->event_times[2],
              Eq(ParseRfc3339("2026-01-01T10:02:00+00:00")));
}

// Verifies that two timestamps representing the same UTC instant but written
// with different timezone offsets parse to identical absl::Time values.
// e.g. 05:30 IST == 00:00 UTC.
TEST(DeserializeCheckpointTest, SameInstantDifferentOffsetsAreEqual) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T05:30:00+05:30",   // IST = 00:00 UTC
                            "2026-01-01T00:00:00+00:00"},  // UTC = 00:00 UTC
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(&parser, "test_query");
  ASSERT_THAT(result, IsOk());

  ASSERT_THAT(result->event_times.size(), Eq(2));
  EXPECT_THAT(result->event_times[0], Eq(result->event_times[1]));
}

// Verifies that two timestamps with the same wall-clock time but different
// timezone offsets parse to *different* absl::Time values.
// e.g. 12:00 UTC and 12:00 PST (UTC-8) are 8 hours apart.
TEST(DeserializeCheckpointTest, SameWallClockDifferentOffsetsAreNotEqual) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user456", kPrivacyIdColumnName));
  tensors.push_back(
      Tensor({"2026-01-01T12:00:00+00:00",   // 12:00 UTC
              "2026-01-01T12:00:00-08:00"},  // 12:00 PST = 20:00 UTC
             "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = DeserializeCheckpoint(&parser, "test_query");
  ASSERT_THAT(result, IsOk());

  ASSERT_THAT(result->event_times.size(), Eq(2));
  EXPECT_NE(result->event_times[0], result->event_times[1]);
}

}  // namespace
}  // namespace confidential_federated_compute::construct_user_session
