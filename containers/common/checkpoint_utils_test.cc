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

#include "containers/common/checkpoint_utils.h"

#include <cstdint>
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

namespace confidential_federated_compute {
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

TEST(CheckpointUtilsTest, GetPrivacyIdValid) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor("user123", kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetPrivacyId(parser);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, Eq("user123"));
}

TEST(CheckpointUtilsTest, GetPrivacyIdMissing) {
  std::vector<Tensor> tensors;
  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetPrivacyId(parser);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kNotFound,
                               HasSubstr("confidential_compute_privacy_id")));
}

TEST(CheckpointUtilsTest, GetPrivacyIdWrongType) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor(int64_t{123}, kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetPrivacyId(parser);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("`confidential_compute_privacy_id` "
                                         "tensor must be a string tensor")));
}

TEST(CheckpointUtilsTest, GetPrivacyIdWrongShape) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({"user1", "user2"}, kPrivacyIdColumnName));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetPrivacyId(parser);
  EXPECT_THAT(
      result,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "`confidential_compute_privacy_id` tensor must be a scalar")));
}

TEST(CheckpointUtilsTest, GetEventTimeValid) {
  std::vector<Tensor> tensors;
  tensors.push_back(Tensor({"2026-01-01T10:00:00Z", "2026-01-01T10:01:00Z"},
                           "my_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetEventTime(parser, "my_query");
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->ToStringVector(),
              ElementsAre("2026-01-01T10:00:00Z", "2026-01-01T10:01:00Z"));
}

TEST(CheckpointUtilsTest, GetEventTimeMissing) {
  std::vector<Tensor> tensors;
  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetEventTime(parser, "my_query");
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("my_query/confidential_compute_event_time")));
}

TEST(CheckpointUtilsTest, GetEventTimeWrongType) {
  std::vector<Tensor> tensors;
  tensors.push_back(
      Tensor({int64_t{123456}}, "my_query/confidential_compute_event_time"));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetEventTime(parser, "my_query");
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`my_query/confidential_compute_event_time` "
                                 "tensor must be a string tensor")));
}

TEST(CheckpointUtilsTest, GetEventTimeWrongShape) {
  std::vector<Tensor> tensors;
  // Create a 2D tensor (1x1) to trigger the wrong shape error.
  auto time_tensor = Tensor::Create(
      DataType::DT_STRING, TensorShape({1, 1}),
      CreateTestData<absl::string_view>({"2026-01-01T10:00:00Z"}),
      "my_query/confidential_compute_event_time");
  CHECK_OK(time_tensor);
  tensors.push_back(*std::move(time_tensor));

  InMemoryCheckpointParser parser(std::move(tensors));
  auto result = GetEventTime(parser, "my_query");
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`my_query/confidential_compute_event_time` "
                                 "tensor must have one dimension")));
}

}  // namespace
}  // namespace confidential_federated_compute
