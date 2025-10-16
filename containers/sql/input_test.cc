// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "containers/sql/input.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/sql/row_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/matchers.h"

namespace confidential_federated_compute::sql {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

class InputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto t1 = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                             CreateTestData<int64_t>({1, 2}), "col1");
    CHECK_OK(t1);
    contents_.push_back(*std::move(t1));
    auto t2 = Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                             CreateTestData<absl::string_view>({"foo", "bar"}),
                             "col2");
    CHECK_OK(t2);
    contents_.push_back(*std::move(t2));

    blob_header_.set_access_policy_node_id(42);
  }
  std::vector<Tensor> contents_;
  fcp::confidentialcompute::BlobHeader blob_header_;
};

TEST_F(InputTest, CreateFromTensors) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
}

TEST_F(InputTest, CreateFromTensorsFailsWithNoColumns) {
  absl::StatusOr<Input> input = Input::CreateFromTensors({}, blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "No columns provided."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithNoRows) {
  contents_.clear();
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({}),
                                      CreateTestData<int64_t>({1}), "col1"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "Column has no rows."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithMultiDimensionalRows) {
  contents_.clear();
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({1, 2}),
                                      CreateTestData<int64_t>({1, 2}), "col1"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "Column has more than one dimension."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithMismatchedRows) {
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({3}),
                                      CreateTestData<int64_t>({1, 2, 3}),
                                      "col3"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(
      input.status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             "All columns must have the same number of rows."));
}

TEST_F(InputTest, GetColumnNames) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetColumnNames(),
            std::vector<std::string>({"col1", "col2"}));
}

TEST_F(InputTest, GetBlobHeader) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_THAT(input->blob_header(), EqualsProto(blob_header_));
}

TEST_F(InputTest, GetRow) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());
  EXPECT_EQ(row->GetColumnCount(), 2);
  EXPECT_EQ(row->GetValue<int64_t>(0), 1);
  EXPECT_EQ(row->GetValue<absl::string_view>(1), "foo");

  absl::StatusOr<RowView> row1 = input->GetRow(1);
  ASSERT_THAT(row1, IsOk());
  EXPECT_EQ(row1->GetValue<int64_t>(0), 2);
  EXPECT_EQ(row1->GetValue<absl::string_view>(1), "bar");
}

TEST_F(InputTest, GetRowCount) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetRowCount(), 2);
}

TEST_F(InputTest, MoveToTensors) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  std::vector<Tensor> tensors = std::move(*input).MoveToTensors();
  ASSERT_EQ(tensors.size(), 2);
  EXPECT_EQ(tensors[0].name(), "col1");
  EXPECT_EQ(tensors[0].dtype(), DataType::DT_INT64);
  EXPECT_EQ(tensors[0].shape(), TensorShape({2}));
  EXPECT_THAT(tensors[0].AsSpan<int64_t>(), testing::ElementsAre(1, 2));
  EXPECT_EQ(tensors[1].name(), "col2");
  EXPECT_EQ(tensors[1].dtype(), DataType::DT_STRING);
  EXPECT_EQ(tensors[1].shape(), TensorShape({2}));
  EXPECT_THAT(tensors[1].AsSpan<absl::string_view>(),
              testing::ElementsAre("foo", "bar"));
}

}  // namespace
}  // namespace confidential_federated_compute::sql
