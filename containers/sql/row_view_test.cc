// Copyright 2025 Google LLC.
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

#include "containers/sql/row_view.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/matchers.h"

namespace confidential_federated_compute::sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::Eq;
using ::testing::IsEmpty;

class RowViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    columns_.push_back(*Tensor::Create(DataType::DT_INT32, TensorShape({3}),
                                       CreateTestData<int32_t>({1, 2, 3}),
                                       "col_int32"));
    columns_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({3}),
                                       CreateTestData<int64_t>({4, 5, 6}),
                                       "col_int64"));
    columns_.push_back(*Tensor::Create(
        DataType::DT_FLOAT, TensorShape({3}),
        CreateTestData<float>({1.1f, 2.2f, 3.3f}), "col_float"));
    columns_.push_back(*Tensor::Create(DataType::DT_DOUBLE, TensorShape({3}),
                                       CreateTestData<double>({4.4, 5.5, 6.6}),
                                       "col_double"));
    columns_.push_back(*Tensor::Create(
        DataType::DT_STRING, TensorShape({3}),
        CreateTestData<absl::string_view>({"foo", "bar", "baz"}),
        "col_string"));
  }

  std::vector<Tensor> columns_;
};

TEST_F(RowViewTest, CreateSuccess) {
  ASSERT_THAT(RowView::CreateFromTensors(columns_, 0), IsOk());
}

TEST_F(RowViewTest, CreateWithEmptyColumnsFails) {
  std::vector<Tensor> empty_columns;
  EXPECT_THAT(RowView::CreateFromTensors(empty_columns, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RowViewTest, CreateWithColumnWithNoRowsFails) {
  std::vector<Tensor> columns;
  columns.push_back(*Tensor::Create(DataType::DT_INT32, TensorShape({0}),
                                    CreateTestData<int32_t>({}), "col_int32"));
  EXPECT_THAT(RowView::CreateFromTensors(columns, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RowViewTest, GetValue) {
  auto row_view = RowView::CreateFromTensors(columns_, 1);
  ASSERT_THAT(row_view, IsOk());
  EXPECT_THAT(row_view->GetValue<int32_t>(0), Eq(2));
  EXPECT_THAT(row_view->GetValue<int64_t>(1), Eq(5));
  EXPECT_THAT(row_view->GetValue<float>(2), Eq(2.2f));
  EXPECT_THAT(row_view->GetValue<double>(3), Eq(5.5));
  EXPECT_THAT(row_view->GetValue<absl::string_view>(4), Eq("bar"));
}

TEST_F(RowViewTest, GetColumnCount) {
  auto row_view = RowView::CreateFromTensors(columns_, 0);
  ASSERT_THAT(row_view, IsOk());
  EXPECT_THAT(row_view->GetColumnCount(), Eq(5));
}

TEST_F(RowViewTest, GetColumnType) {
  auto row_view = RowView::CreateFromTensors(columns_, 0);
  ASSERT_THAT(row_view, IsOk());
  EXPECT_THAT(row_view->GetColumnType(0), Eq(DataType::DT_INT32));
  EXPECT_THAT(row_view->GetColumnType(1), Eq(DataType::DT_INT64));
  EXPECT_THAT(row_view->GetColumnType(2), Eq(DataType::DT_FLOAT));
  EXPECT_THAT(row_view->GetColumnType(3), Eq(DataType::DT_DOUBLE));
  EXPECT_THAT(row_view->GetColumnType(4), Eq(DataType::DT_STRING));
}

TEST_F(RowViewTest, MismatchedTypeDeathTest) {
  auto row_view = RowView::CreateFromTensors(columns_, 0);
  ASSERT_THAT(row_view, IsOk());
  EXPECT_DEATH(row_view->GetValue<absl::string_view>(0),
               "Incompatible tensor dtype");
}

TEST_F(RowViewTest, RowIndexOutOfBounds) {
  EXPECT_THAT(RowView::CreateFromTensors(columns_, 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace confidential_federated_compute::sql
