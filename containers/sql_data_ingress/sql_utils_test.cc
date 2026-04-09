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

#include "containers/sql_data_ingress/sql_utils.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/sql/input.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::sql_data_ingress {

namespace {

using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowSet;
using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::TableSchema;
using ::tensorflow_federated::aggregation::AggVector;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::PrintToString;
using ::testing::UnorderedElementsAre;

// Matcher for Tensor name
MATCHER_P(TensorHasName, expected_name,
          std::string("has '") + expected_name + "'") {
  *result_listener << "whose name is '" << arg.name() << "'";
  return arg.name() == expected_name;
}

// Converts a potentially sparse tensor to a flat vector of tensor values.
template <typename T>
std::vector<T> TensorValuesToVector(const Tensor& arg) {
  std::vector<T> vec(arg.num_elements());
  if (arg.num_elements() > 0) {
    AggVector<T> agg_vector = arg.AsAggVector<T>();
    for (auto [i, v] : agg_vector) {
      vec[i] = v;
    }
  }
  return vec;
}

MATCHER_P(Int64TensorEq, expected_contents,
          std::string("tensor is ") + PrintToString(expected_contents)) {
  if (arg.dtype() != DataType::DT_INT64) {
    return false;
  }
  std::vector<int64_t> flat_vector = TensorValuesToVector<int64_t>(arg);
  *result_listener << "tensor is '" << PrintToString(flat_vector) << "'";
  return flat_vector == expected_contents;
}

TEST(CreateRowLocationsForAllRowsTest, ZeroRowsReturnsEmpty) {
  EXPECT_THAT(CreateRowLocationsForAllRows(0), ::testing::IsEmpty());
}

TEST(CreateRowLocationsForAllRowsTest, ReturnsCorrectLocations) {
  auto locations = CreateRowLocationsForAllRows(3);
  EXPECT_THAT(locations, ElementsAre(FieldsAre(0, 0, 0), FieldsAre(0, 0, 1),
                                     FieldsAre(0, 0, 2)));
}

TEST(ExecuteClientQueryTest, SimpleQuerySucceeds) {
  ASSERT_TRUE(SqliteAdapter::Initialize().ok());
  SqlConfiguration config;
  config.query = "SELECT key, val * 2 AS val FROM input";
  config.input_schema = PARSE_TEXT_PROTO(R"pb(
    name: "input"
    column { name: "key" type: INT64 }
    column { name: "val" type: INT64 }
    create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER)"
  )pb");
  TableSchema output_schema = PARSE_TEXT_PROTO(R"pb(
    column { name: "key" type: INT64 }
    column { name: "val" type: INT64 }
  )pb");
  config.output_columns.CopyFrom(output_schema.column());

  std::vector<Tensor> columns;
  auto key_tensor_or = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                                      CreateTestData<int64_t>({1, 2}), "key");
  ASSERT_TRUE(key_tensor_or.ok());
  columns.push_back(std::move(*key_tensor_or));

  auto val_tensor_or = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                                      CreateTestData<int64_t>({10, 20}), "val");
  ASSERT_TRUE(val_tensor_or.ok());
  columns.push_back(std::move(*val_tensor_or));

  BlobHeader dummy_header;
  auto input_or = Input::CreateFromTensors(std::move(columns), dummy_header);
  ASSERT_TRUE(input_or.ok());
  Input input = std::move(*input_or);

  std::vector<RowLocation> locations =
      CreateRowLocationsForAllRows(input.GetRowCount());
  absl::Span<Input> storage = absl::MakeSpan(&input, 1);
  auto row_set_or = RowSet::Create(locations, storage);
  ASSERT_TRUE(row_set_or.ok());

  auto result_tensors_or = ExecuteClientQuery(config, std::move(*row_set_or));
  ASSERT_TRUE(result_tensors_or.ok());
  std::vector<Tensor> result_tensors = std::move(*result_tensors_or);

  EXPECT_THAT(
      result_tensors,
      UnorderedElementsAre(
          ::testing::AllOf(TensorHasName("key"),
                           Int64TensorEq(std::vector<int64_t>{1, 2})),
          ::testing::AllOf(TensorHasName("val"),
                           Int64TensorEq(std::vector<int64_t>{20, 40}))));
}

TEST(ExecuteClientQueryTest, QueryOnNonexistentColumnFails) {
  ASSERT_TRUE(SqliteAdapter::Initialize().ok());
  SqlConfiguration config;
  config.query = "SELECT non_existent_col FROM input";
  config.input_schema = PARSE_TEXT_PROTO(R"pb(
    name: "input"
    column { name: "key" type: INT64 }
    create_table_sql: "CREATE TABLE input (key INTEGER)"
  )pb");
  TableSchema output_schema = PARSE_TEXT_PROTO(R"pb(
    column { name: "non_existent_col" type: INT64 }
  )pb");
  config.output_columns.CopyFrom(output_schema.column());

  std::vector<Tensor> columns;
  auto key_tensor_or = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                                      CreateTestData<int64_t>({1, 2}), "key");
  ASSERT_TRUE(key_tensor_or.ok());
  columns.push_back(std::move(*key_tensor_or));

  BlobHeader dummy_header;
  auto input_or = Input::CreateFromTensors(std::move(columns), dummy_header);
  ASSERT_TRUE(input_or.ok());
  Input input = std::move(*input_or);

  std::vector<RowLocation> locations =
      CreateRowLocationsForAllRows(input.GetRowCount());
  absl::Span<Input> storage = absl::MakeSpan(&input, 1);
  auto row_set_or = RowSet::Create(locations, storage);
  ASSERT_TRUE(row_set_or.ok());

  auto result = ExecuteClientQuery(config, std::move(*row_set_or));
  EXPECT_FALSE(result.ok());
}

}  // namespace

}  // namespace confidential_federated_compute::sql_data_ingress
