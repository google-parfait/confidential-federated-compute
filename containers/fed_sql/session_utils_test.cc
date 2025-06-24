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
#include "containers/fed_sql/session_utils.h"

#include <filesystem>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "containers/sql/sqlite_adapter.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {
using ::confidential_federated_compute::fed_sql::testing::
    BuildFedSqlGroupByCheckpoint;
using ::confidential_federated_compute::sql::TensorColumn;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::tensorflow_federated::aggregation::AggVector;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::Each;
using ::testing::PrintToString;
using ::testing::UnorderedElementsAre;

MATCHER_P(TensorColumnHasName, expected_name,
          std::string("has column schema name '") + expected_name + "'") {
  *result_listener << "whose column schema name is '"
                   << arg.column_schema_.name() << "'";
  return arg.column_schema_.name() == expected_name;
}

MATCHER_P(TensorHasNumElements, expected_num_elements,
          std::string("tensor has ") + PrintToString(expected_num_elements) +
              " elements") {
  const size_t actual_num_elements = arg.tensor_.num_elements();
  if (actual_num_elements != expected_num_elements) {
    *result_listener << "tensor actually has " << actual_num_elements
                     << " elements";
    return false;
  }
  return true;
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

MATCHER_P(Int64TensorColumnEq, expected_contents,
          std::string("has tensor ") + PrintToString(expected_contents)) {
  if (arg.tensor_.dtype() != DataType::DT_INT64) {
    return false;
  }
  std::vector<int64_t> flat_vector = TensorValuesToVector<int64_t>(arg.tensor_);
  *result_listener << "whose tensor is '" << PrintToString(flat_vector) << "'";
  return flat_vector == expected_contents;
}

MATCHER_P(StringTensorColumnEq, expected_contents,
          std::string("has tensor ") + PrintToString(expected_contents)) {
  if (arg.tensor_.dtype() != DataType::DT_STRING) {
    return false;
  }
  std::vector<absl::string_view> flat_vector =
      TensorValuesToVector<absl::string_view>(arg.tensor_);
  *result_listener << "whose tensor is '" << PrintToString(flat_vector) << "'";
  return flat_vector == expected_contents;
}

TEST(DeserializeTest, DeserializeSucceedsWithoutInferenceConfig) {
  TableSchema schema = PARSE_TEXT_PROTO(R"pb(
    name: "input"
    column { name: "key" type: INT64 }
    column { name: "val" type: INT64 }
    create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER)"
  )pb");

  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1}, "key", "val");
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(data));
  auto deserialized_result = Deserialize(schema, parser->get());
  EXPECT_THAT(deserialized_result, IsOk());
  EXPECT_EQ(deserialized_result->size(), 2);
  EXPECT_THAT(*deserialized_result,
              UnorderedElementsAre(TensorColumnHasName("key"),
                                   TensorColumnHasName("val")));
  EXPECT_THAT(*deserialized_result, Each(TensorHasNumElements(1)));
  EXPECT_THAT(
      *deserialized_result,
      UnorderedElementsAre(Int64TensorColumnEq(std::vector<int64_t>{8}),
                           Int64TensorColumnEq(std::vector<int64_t>{1})));
}

std::string BuildInferenceCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<absl::string_view> inference_input_col_values,
    const std::string& key_col_name,
    const std::string& inference_input_col_name) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<uint64_t>(key_col_values));
  absl::StatusOr<Tensor> inference_col = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(inference_input_col_values.size())}),
      CreateTestData<absl::string_view>(inference_input_col_values));
  CHECK_OK(key);
  CHECK_OK(inference_col);
  CHECK_OK(ckpt_builder->Add(key_col_name, *key));
  CHECK_OK(ckpt_builder->Add(inference_input_col_name, *inference_col));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

TEST(DeserializeTest, DeserializeSucceedsWithInferenceConfig) {
  // The serialized checkpoint contains columns "key" and "val". The TableSchema
  // contains the "key" column and the inference config contains the "val" input
  // column.
  TableSchema schema = PARSE_TEXT_PROTO(R"pb(
    name: "input"
    column { name: "key" type: INT64 }
    column { name: "topic" type: STRING }
    create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER)"
  )pb");

  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config { input_column_name: "val" output_column_name: "topic" }
      }
    }
  )pb");
  std::string data = BuildInferenceCheckpoint({8}, {"abc"}, "key", "val");
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(data));
  auto deserialized_result =
      Deserialize(schema, parser->get(), inference_configuration);
  EXPECT_THAT(deserialized_result, IsOk());
  EXPECT_EQ(deserialized_result->size(), 2);
  EXPECT_THAT(*deserialized_result,
              UnorderedElementsAre(TensorColumnHasName("key"),
                                   TensorColumnHasName("val")));
  EXPECT_THAT(*deserialized_result, Each(TensorHasNumElements(1)));
  EXPECT_THAT(*deserialized_result,
              UnorderedElementsAre(
                  Int64TensorColumnEq(std::vector<int64_t>{8}),
                  StringTensorColumnEq(std::vector<absl::string_view>{"abc"})));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
