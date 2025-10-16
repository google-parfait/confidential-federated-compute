// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law
// or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "containers/sql/row_set.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/sql/input.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace confidential_federated_compute::sql {
namespace {

using ::absl_testing::IsOk;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

class TensorRowSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create two Input objects for testing.
    // Input 1 has 2 rows and 2 columns (int64_t, string).
    std::vector<Tensor> contents1;
    auto t1_1 = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                               CreateTestData<int64_t>({1, 2}), "int_col");
    CHECK_OK(t1_1);
    contents1.push_back(*std::move(t1_1));
    auto t1_2 = Tensor::Create(
        DataType::DT_STRING, TensorShape({2}),
        CreateTestData<absl::string_view>({"foo", "bar"}), "string_col");
    CHECK_OK(t1_2);
    contents1.push_back(*std::move(t1_2));
    absl::StatusOr<Input> input1 =
        Input::CreateFromTensors(std::move(contents1), {});
    CHECK_OK(input1);

    // Input 2 has 3 rows and 2 columns (int64_t, string).
    std::vector<Tensor> contents2;
    auto t2_1 = Tensor::Create(DataType::DT_INT64, TensorShape({3}),
                               CreateTestData<int64_t>({3, 4, 5}), "int_col");
    CHECK_OK(t2_1);
    contents2.push_back(*std::move(t2_1));
    auto t2_2 = Tensor::Create(
        DataType::DT_STRING, TensorShape({3}),
        CreateTestData<absl::string_view>({"baz", "qux", "quux"}),
        "string_col");
    CHECK_OK(t2_2);
    contents2.push_back(*std::move(t2_2));
    absl::StatusOr<Input> input2 =
        Input::CreateFromTensors(std::move(contents2), {});
    CHECK_OK(input2);

    inputs_.push_back(*std::move(input1));
    inputs_.push_back(*std::move(input2));
  }

  std::vector<std::vector<std::string>> CollectRows(const RowSet& set) {
    std::vector<std::vector<std::string>> result;
    for (const RowView& row : set) {
      std::vector<std::string> row_strings;
      for (int i = 0; i < row.GetColumnCount(); ++i) {
        switch (row.GetColumnType(i)) {
          case DataType::DT_INT64:
            row_strings.push_back(std::to_string(row.GetValue<int64_t>(i)));
            break;
          case DataType::DT_STRING:
            row_strings.push_back(
                std::string(row.GetValue<absl::string_view>(i)));
            break;
          default:
            // All types used in this test are handled above.
            break;
        }
      }
      result.push_back(row_strings);
    }
    return result;
  }

  std::vector<Input> inputs_;
};

TEST_F(TensorRowSetTest, EmptySet) {
  std::vector<RowLocation> locations;
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set), IsEmpty());
}

TEST_F(TensorRowSetTest, SingleRowFromFirstInput) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 1}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set), ElementsAre(ElementsAre("2", "bar")));
}

TEST_F(TensorRowSetTest, SingleRowFromSecondInput) {
  std::vector<RowLocation> locations = {{.input_index = 1, .row_index = 2}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set), ElementsAre(ElementsAre("5", "quux")));
}

TEST_F(TensorRowSetTest, MultipleRowsFromSingleInput) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 0},
                                        {.input_index = 0, .row_index = 1}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set),
              ElementsAre(ElementsAre("1", "foo"), ElementsAre("2", "bar")));
}

TEST_F(TensorRowSetTest, MultipleRowsFromMultipleInputs) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 1},
                                        {.input_index = 1, .row_index = 0},
                                        {.input_index = 1, .row_index = 2}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set),
              ElementsAre(ElementsAre("2", "bar"), ElementsAre("3", "baz"),
                          ElementsAre("5", "quux")));
}

TEST_F(TensorRowSetTest, NonSequentialAccess) {
  // Access rows in a non-sequential order to test that the iterator
  // correctly follows the locations vector.
  std::vector<RowLocation> locations = {{.input_index = 1, .row_index = 2},
                                        {.input_index = 0, .row_index = 0},
                                        {.input_index = 1, .row_index = 1}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  EXPECT_THAT(CollectRows(*set),
              ElementsAre(ElementsAre("5", "quux"), ElementsAre("1", "foo"),
                          ElementsAre("4", "qux")));
}

TEST_F(TensorRowSetTest, CreateFailsWithDifferentColumnNames) {
  // Create two Input objects that have different column names.
  std::vector<Tensor> contents1;
  auto t1 = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                           CreateTestData<int64_t>({1, 2}), "int_col");
  ASSERT_THAT(t1, IsOk());
  contents1.push_back(*std::move(t1));
  absl::StatusOr<Input> input1 =
      Input::CreateFromTensors(std::move(contents1), {});
  ASSERT_THAT(input1, IsOk());

  std::vector<Tensor> contents2;
  auto t2 = Tensor::Create(
      DataType::DT_STRING, TensorShape({3}),
      CreateTestData<absl::string_view>({"baz", "qux", "quux"}), "string_col");
  ASSERT_THAT(t2, IsOk());
  contents2.push_back(*std::move(t2));
  absl::StatusOr<Input> input2 =
      Input::CreateFromTensors(std::move(contents2), {});
  ASSERT_THAT(input2, IsOk());

  std::vector<Input> inputs;
  inputs.push_back(*std::move(input1));
  inputs.push_back(*std::move(input2));

  std::vector<RowLocation> locations;
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs);
  ASSERT_FALSE(set.ok());
  EXPECT_EQ(set.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(TensorRowSetTest, IteratorEquality) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 0},
                                        {.input_index = 0, .row_index = 1}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  auto it1 = set->begin();
  auto it2 = set->begin();
  auto end = set->end();

  EXPECT_TRUE(it1 == it2);
  EXPECT_FALSE(it1 == end);
  EXPECT_TRUE(it1 != end);

  ++it1;
  EXPECT_FALSE(it1 == it2);
  EXPECT_TRUE(it1 != it2);

  ++it2;
  EXPECT_TRUE(it1 == it2);

  ++it1;
  EXPECT_TRUE(it1 == end);
}

TEST_F(TensorRowSetTest, GetColumnNames) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 0}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  auto column_names = set->GetColumnNames();
  ASSERT_TRUE(column_names.ok());
  EXPECT_THAT(*column_names, ElementsAre("int_col", "string_col"));
}

TEST_F(TensorRowSetTest, GetColumnNamesForSetWithEmptyLocations) {
  std::vector<RowLocation> locations;
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  auto column_names = set->GetColumnNames();
  ASSERT_TRUE(column_names.ok());
  EXPECT_THAT(*column_names, ElementsAre("int_col", "string_col"));
}

TEST_F(TensorRowSetTest, GetColumnNamesForSetWithEmptyStorage) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 0}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, {});
  ASSERT_TRUE(set.ok());
  auto column_names = set->GetColumnNames();
  ASSERT_TRUE(column_names.ok());
  EXPECT_THAT(*column_names, IsEmpty());
}

TEST_F(TensorRowSetTest, DereferenceInvalidRowDeathTest) {
  std::vector<RowLocation> locations = {{.input_index = 0, .row_index = 2}};
  absl::StatusOr<RowSet> set = RowSet::Create(locations, inputs_);
  ASSERT_TRUE(set.ok());
  auto it = set->begin();
  EXPECT_DEATH(*it, "Check failed");
}

}  // namespace
}  // namespace confidential_federated_compute::sql
