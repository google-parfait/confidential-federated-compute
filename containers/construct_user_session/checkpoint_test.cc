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

#include "containers/construct_user_session/checkpoint.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
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
using ::testing::Pair;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

template <typename NameMatcher, typename ValuesMatcher>
auto ColumnTensorIs(NameMatcher name_matcher, ValuesMatcher values_matcher) {
  return Pair(name_matcher, Property(&Tensor::ToStringVector, values_matcher));
}

TEST(CheckpointTest, ValidCheckpoint) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00", "bad-timestamp"},
                           "test_query/confidential_compute_event_time"));
  tensors.push_back(Tensor({"valA1", "valA2"}, "test_query/colA"));
  tensors.push_back(Tensor({"valB1", "valB2"}, "colB"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id(), Eq("user123"));

  EXPECT_THAT(
      result->column_tensors(),
      UnorderedElementsAre(
          ColumnTensorIs(
              "test_query/confidential_compute_event_time",
              ElementsAre("2026-01-01T10:00:00+00:00", "bad-timestamp")),
          ColumnTensorIs("test_query/colA", ElementsAre("valA1", "valA2")),
          ColumnTensorIs("colB", ElementsAre("valB1", "valB2"))));
}

TEST(CheckpointTest, MissingPrivacyIdFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_privacy_id")));
}

TEST(CheckpointTest, MissingEventTimeFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_event_time")));
}

TEST(CheckpointTest, MultidimensionalColumnTensorFails) {
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
  auto result = Checkpoint::Create(parser, "test_query");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Column tensor `test_query/colA` must "
                                         "have one dimension.")));
}

TEST(CheckpointTest, MismatchedRowCountsFails) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(
      Tensor({"2026-01-01T10:00:00+00:00", "2026-01-01T10:01:00+00:00"},
             "test_query/confidential_compute_event_time"));
  tensors.push_back(
      Tensor({"valA1"}, "test_query/colA"));  // 1 row instead of 2.

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Column tensor `test_query/colA` has 1 rows, "
                                 "expected 2 matching event time tensor.")));
}

// Verifies that a checkpoint with no client data columns (only metadata and
// event time) deserializes successfully with column_tensors containing only the
// event time.
TEST(CheckpointTest, NoClientDataColumnsSucceeds) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user789", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-03-01T08:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_THAT(result->privacy_id(), Eq("user789"));
  // Only the event time tensor is in column_tensors.
  EXPECT_THAT(result->column_tensors(),
              UnorderedElementsAre(
                  ColumnTensorIs("test_query/confidential_compute_event_time",
                                 ElementsAre("2026-03-01T08:00:00+00:00"))));
}

TEST(CheckpointTest, PrivacyIdGetterSucceeds) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("precise_user_id_42", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  EXPECT_EQ(result->privacy_id(), "precise_user_id_42");
}

// Verifies that take_privacy_id_tensor() moves the tensor out and that the
// returned tensor is a valid scalar DT_STRING tensor.
TEST(CheckpointTest, TakePrivacyIdTensorMoveSemantics) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user_to_move", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  // Move the privacy ID tensor out.
  Tensor moved_tensor = result->take_privacy_id_tensor();
  EXPECT_TRUE(moved_tensor.is_scalar());
  EXPECT_EQ(moved_tensor.dtype(), DataType::DT_STRING);
  EXPECT_EQ(moved_tensor.AsScalar<absl::string_view>(), "user_to_move");
}

// Verifies that column_tensors() does not include the privacy ID tensor.
TEST(CheckpointTest, ColumnTensorsExcludePrivacyId) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));
  tensors.push_back(Tensor({"2026-01-01T10:00:00+00:00"},
                           "test_query/confidential_compute_event_time"));
  tensors.push_back(Tensor({"val1"}, "test_query/colA"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = Checkpoint::Create(parser, "test_query");
  ASSERT_THAT(result, IsOk());

  // Privacy ID should not be in column_tensors.
  EXPECT_EQ(result->column_tensors().find(std::string(kPrivacyIdColumnName)),
            result->column_tensors().end());
}

}  // namespace
}  // namespace confidential_federated_compute::construct_user_session
