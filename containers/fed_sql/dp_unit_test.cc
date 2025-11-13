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
#include "containers/fed_sql/dp_unit.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/optional.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "containers/sql/input.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/confidentialcompute/constants.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fed_sql::testing::CreateStringTestData;
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::RepeatedPtrField;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::IsEmpty;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

Configuration DefaultConfiguration() {
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key_out"
        dtype: DT_INT64
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "val"
            dtype: DT_INT64
            shape {}
          }
        }
        output_tensors {
          name: "val_out"
          dtype: DT_INT64
          shape {}
        }
      }
    }
  )pb");
}

class DpUnitCommitTest : public Test {
 protected:
  void SetUp() override {
    CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
    absl::StatusOr<std::unique_ptr<CheckpointAggregator>> aggregator =
        CheckpointAggregator::Create(DefaultConfiguration());
    CHECK_OK(aggregator.status());
    aggregator_ = *std::move(aggregator);

    // SUM(val) + 1 so we can tell how many times the query was run.
    sql_config_.query =
        "SELECT key, SUM(val) + 1 AS val FROM input GROUP BY key";
    TableSchema input_schema = PARSE_TEXT_PROTO(R"pb(
      name: "input"
      column { name: "key" type: INT64 }
      column { name: "val" type: INT64 }
      create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER)"
    )pb");
    sql_config_.input_schema = input_schema;
    RepeatedPtrField<ColumnSchema> output_columns;
    *output_columns.Add() = PARSE_TEXT_PROTO(R"pb(name: "key" type: INT64)pb");
    *output_columns.Add() = PARSE_TEXT_PROTO(R"pb(name: "val" type: INT64)pb");
    sql_config_.output_columns = output_columns;

    id1_ = "1";
    id2_ = "2";  // Different ID

    // Create a windowing schedule.
    schedule_ = PARSE_TEXT_PROTO(R"pb(
      size { size: 1 unit: HOURS }
      shift { size: 1 unit: HOURS }
      start_date { year: 2025 month: 1 day: 1 }
    )pb");
  }

  // Member variables.
  std::unique_ptr<CheckpointAggregator> aggregator_;
  SqlConfiguration sql_config_;
  fcp::confidentialcompute::WindowingSchedule::CivilTimeWindowSchedule
      schedule_;
  std::string id1_;
  std::string id2_;
};

// A helper to build a vector of Tensors for DP unit tests.
//
// This function constructs `Tensor` objects from the provided column data.
// Tensors are named "key", "val", the value of `kEventTimeColumnName`, and the
// names provided in the `dp_columns` map. All input vectors must have the
// same size.
absl::StatusOr<std::vector<Tensor>> BuildTensors(
    std::vector<int64_t> keys, std::vector<int64_t> vals,
    std::vector<std::string> event_times,
    absl::flat_hash_map<std::string, std::vector<int64_t>> dp_columns) {
  if (keys.size() != vals.size() || keys.size() != event_times.size()) {
    return absl::InvalidArgumentError("Input vectors must have the same size.");
  }
  for (const auto& [name, values] : dp_columns) {
    if (values.size() != keys.size()) {
      return absl::InvalidArgumentError(
          "DP column vectors must have the same size as other inputs.");
    }
  }

  size_t num_rows = keys.size();
  std::vector<Tensor> tensors;

  // Key tensor
  auto key_data = std::make_unique<MutableVectorData<int64_t>>(std::move(keys));
  FCP_ASSIGN_OR_RETURN(
      Tensor key_tensor,
      Tensor::Create(DataType::DT_INT64, TensorShape({(int64_t)num_rows}),
                     std::move(key_data)));
  key_tensor.set_name("key");
  tensors.push_back(std::move(key_tensor));

  // Val tensor
  auto val_data = std::make_unique<MutableVectorData<int64_t>>(std::move(vals));
  FCP_ASSIGN_OR_RETURN(
      Tensor val_tensor,
      Tensor::Create(DataType::DT_INT64, TensorShape({(int64_t)num_rows}),
                     std::move(val_data)));
  val_tensor.set_name("val");
  tensors.push_back(std::move(val_tensor));

  // Event time tensor
  FCP_ASSIGN_OR_RETURN(
      Tensor event_times_tensor,
      Tensor::Create(DataType::DT_STRING, {(int64_t)num_rows},
                     CreateStringTestData(std::move(event_times))));
  event_times_tensor.set_name(fcp::confidential_compute::kEventTimeColumnName);
  tensors.push_back(std::move(event_times_tensor));

  // DP column tensors
  for (auto& [name, values] : dp_columns) {
    auto col_data =
        std::make_unique<MutableVectorData<int64_t>>(std::move(values));
    FCP_ASSIGN_OR_RETURN(Tensor dp_column_tensor,
                         Tensor::Create(DataType::DT_INT64, {(int64_t)num_rows},
                                        std::move(col_data)));
    dp_column_tensor.set_name(name);
    tensors.push_back(std::move(dp_column_tensor));
  }

  return tensors;
}

// A helper to build a privacy ID tensor.
absl::StatusOr<Tensor> CreatePrivacyIdTensor(std::string id) {
  auto privacy_id_data = std::make_unique<MutableStringData>(1);
  privacy_id_data->Add(std::move(id));
  FCP_ASSIGN_OR_RETURN(
      Tensor privacy_id_tensor,
      Tensor::Create(DataType::DT_STRING, {}, std::move(privacy_id_data)));
  privacy_id_tensor.set_name(kPrivacyIdColumnName);
  return privacy_id_tensor;
}

TEST_F(DpUnitCommitTest, SuccessWithSingleColumn) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input 1 (privacy_id = 1) contains:
  // +-----+-----+-----------------------------+-------+
  // | key | val | event_time                  | unit1 |
  // +-----+-----+-----------------------------+-------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00   |   0   |
  // |  2  | 200 | 2025-01-01T12:00:00+00:00   |   1   |
  // |  1  | 300 | 2025-01-01T12:00:00+00:00   |   0   |
  // +-----+-----+-----------------------------+-------+
  //
  // Input 2 (privacy_id = 1) contains:
  // +-----+-----+-----------------------------+-------+
  // | key | val | event_time                  | unit1 |
  // +-----+-----+-----------------------------+-------+
  // |  1  | 400 | 2025-01-01T12:00:00+00:00   |   0   |
  // |  3  | 500 | 2025-01-01T12:00:00+00:00   |   1   |
  // +-----+-----+-----------------------------+-------+
  DpUnitParameters dp_unit_parameters;
  dp_unit_parameters.column_names.push_back("unit1");
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1, 2, 1}, {100, 200, 300},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00",
                    "2025-01-01T12:00:00+00:00"},
                   {{"unit1", {0, 1, 0}}});
  ASSERT_THAT(tensors1, IsOk());

  absl::StatusOr<std::vector<Tensor>> tensors2 =
      BuildTensors({1, 3}, {400, 500},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00"},
                   {{"unit1", {0, 1}}});
  ASSERT_THAT(tensors2, IsOk());

  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());
  absl::StatusOr<Tensor> privacy_id_tensor2 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor2, IsOk());
  absl::StatusOr<Input> input2 = Input::CreateFromTensors(
      *std::move(tensors2), {}, *std::move(privacy_id_tensor2));
  ASSERT_THAT(input2, IsOk());

  // For each DP unit, we run the SQL query `SELECT key, SUM(val) + 1 AS val
  // FROM input GROUP BY key` on its subset of rows.
  // Since both inputs share a privacy ID and event times within the same
  // window, rows from both inputs are grouped into DP units based on the value
  // of `unit1`. The SQL query should be run once for all data in a DP unit.
  //
  // DP Unit (unit1 = 0) contains:
  // - from Input 1: (1, 100), (1, 300)
  // - from Input 2: (1, 400)
  // The SQL query on this combined data results in: (1, 801)
  //
  // DP Unit (unit1 = 1) contains:
  // - from Input 1: (2, 200)
  // - from Input 2: (3, 500)
  // The SQL query on this combined data results in: (2, 201), (3, 501)

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input2)), IsOk());
  ASSERT_THAT(input_processor.CommitRowsGroupingByDpUnit(),
              IsOkAndHolds(IsEmpty()));

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      checkpoint_builder = builder_factory.Create();
  EXPECT_THAT(aggregator_->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());

  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_THAT(parser, IsOk());
  auto key_out = (*parser)->GetTensor("key_out");
  ASSERT_THAT(key_out, IsOk());
  EXPECT_THAT(key_out->AsSpan<int64_t>(), UnorderedElementsAre(1, 2, 3));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  // Final aggregated result:
  // +---------+---------+
  // | key_out | val_out |
  // +---------+---------+
  // |    1    |   801   |
  // |    2    |   201   |
  // |    3    |   501   |
  // +---------+---------+
  // For key 1, the data from two inputs is combined into a single DP unit, so
  // the SQL query is run once. `SUM(100, 300, 400) + 1 = 801`.
  EXPECT_THAT(val_out->AsSpan<int64_t>(), UnorderedElementsAre(801, 201, 501));
}

TEST_F(DpUnitCommitTest, SuccessEmptyDpUnitColumns) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input 1 (privacy_id = 1):
  // +-----+-----+---------------------------+
  // | key | val | event_time                |
  // +-----+-----+---------------------------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00 |
  // |  2  | 200 | 2025-01-01T12:00:00+00:00 |
  // |  1  | 300 | 2025-01-01T12:00:00+00:00 |
  // +-----+-----+---------------------------+
  //
  // Input 2 (privacy_id = 2):
  // +-----+-----+---------------------------+
  // | key | val | event_time                |
  // +-----+-----+---------------------------+
  // |  1  | 400 | 2025-01-01T13:00:00+00:00 |
  // |  3  | 500 | 2025-01-01T13:00:00+00:00 |
  // +-----+-----+---------------------------+
  // This test does not use any DP columns.
  DpUnitParameters dp_unit_parameters;
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;

  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1, 2, 1}, {100, 200, 300},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00",
                    "2025-01-01T12:00:00+00:00"},
                   {});
  ASSERT_THAT(tensors1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors2 = BuildTensors(
      {1, 3}, {400, 500},
      {"2025-01-01T13:00:00+00:00", "2025-01-01T13:00:00+00:00"}, {});
  ASSERT_THAT(tensors2, IsOk());

  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());
  absl::StatusOr<Tensor> privacy_id_tensor2 = CreatePrivacyIdTensor(id2_);
  ASSERT_THAT(privacy_id_tensor2, IsOk());
  absl::StatusOr<Input> input2 = Input::CreateFromTensors(
      *std::move(tensors2), {}, *std::move(privacy_id_tensor2));
  ASSERT_THAT(input2, IsOk());

  // For each DP unit, we run the SQL query `SELECT key, SUM(val) + 1 AS val
  // FROM input GROUP BY key` on its subset of rows.
  // With no DP columns, the data is still split into two DP units because the
  // inputs have different privacy IDs and event times that fall into different
  // time windows.
  // DP Unit 1 (from input 1):
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 100 |
  // |  2  | 200 |
  // |  1  | 300 |
  // +-----+-----+
  // Output of SQL query: (1, 401), (2, 201)
  //
  // DP Unit 2 (from input 2):
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 400 |
  // |  3  | 500 |
  // +-----+-----+
  // Output of SQL query: (1, 401), (3, 501)

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input2)), IsOk());
  ASSERT_THAT(input_processor.CommitRowsGroupingByDpUnit(),
              IsOkAndHolds(IsEmpty()));

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      checkpoint_builder = builder_factory.Create();
  EXPECT_THAT(aggregator_->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());

  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_THAT(parser, IsOk());
  auto key_out = (*parser)->GetTensor("key_out");
  ASSERT_THAT(key_out, IsOk());
  EXPECT_THAT(key_out->AsSpan<int64_t>(), UnorderedElementsAre(1, 2, 3));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  // Final aggregated result:
  // +---------+---------+
  // | key_out | val_out |
  // +---------+---------+
  // |    1    |   802   |
  // |    2    |   201   |
  // |    3    |   501   |
  // +---------+---------+
  // We expect the SQL query to be run twice for key 1 since it's present in two
  // DP units. The SQL query adds 1 each time it's executed.
  EXPECT_THAT(val_out->AsSpan<int64_t>(), UnorderedElementsAre(802, 201, 501));
}

TEST_F(DpUnitCommitTest, SuccessMultipleDpUnitColumns) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input 1 (privacy_id = 1):
  // +-----+-----+---------------------------+-------+-------+
  // | key | val | event_time                | unit1 | unit2 |
  // +-----+-----+---------------------------+-------+-------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00 |   0   |   0   |
  // |  2  | 200 | 2025-01-01T12:00:00+00:00 |   0   |   1   |
  // |  1  | 300 | 2025-01-01T12:00:00+00:00 |   1   |   0   |
  // +-----+-----+---------------------------+-------+-------+
  //
  // Input 2 (privacy_id = 1):
  // +-----+-----+---------------------------+-------+-------+
  // | key | val | event_time                | unit1 | unit2 |
  // +-----+-----+---------------------------+-------+-------+
  // |  1  | 400 | 2025-01-01T12:00:00+00:00 |   0   |   0   |
  // |  3  | 500 | 2025-01-01T12:00:00+00:00 |   1   |   1   |
  // +-----+-----+---------------------------+-------+-------+
  DpUnitParameters dp_unit_parameters;
  dp_unit_parameters.column_names.push_back("unit1");
  dp_unit_parameters.column_names.push_back("unit2");
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1, 2, 1}, {100, 200, 300},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00",
                    "2025-01-01T12:00:00+00:00"},
                   {{"unit1", {0, 0, 1}}, {"unit2", {0, 1, 0}}});
  ASSERT_THAT(tensors1, IsOk());

  absl::StatusOr<std::vector<Tensor>> tensors2 =
      BuildTensors({1, 3}, {400, 500},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00"},
                   {{"unit1", {0, 1}}, {"unit2", {0, 1}}});
  ASSERT_THAT(tensors2, IsOk());

  // Use same privacy ID for both inputs.
  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());
  absl::StatusOr<Tensor> privacy_id_tensor2 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor2, IsOk());
  absl::StatusOr<Input> input2 = Input::CreateFromTensors(
      *std::move(tensors2), {}, *std::move(privacy_id_tensor2));
  ASSERT_THAT(input2, IsOk());

  // With same privacy ID and event time, DP units are determined by unit1,
  // unit2.
  // DP Unit (0, 0): (1, 100), (1, 400). SQL -> (1, 501)
  // DP Unit (0, 1): (2, 200). SQL -> (2, 201)
  // DP Unit (1, 0): (1, 300). SQL -> (1, 301)
  // DP Unit (1, 1): (3, 500). SQL -> (3, 501)
  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input2)), IsOk());
  ASSERT_THAT(input_processor.CommitRowsGroupingByDpUnit(),
              IsOkAndHolds(IsEmpty()));

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      checkpoint_builder = builder_factory.Create();
  EXPECT_THAT(aggregator_->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());

  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_THAT(parser, IsOk());
  auto key_out = (*parser)->GetTensor("key_out");
  ASSERT_THAT(key_out, IsOk());
  EXPECT_THAT(key_out->AsSpan<int64_t>(), UnorderedElementsAre(1, 2, 3));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  // Final aggregated result:
  // key 1: 501 (from 0,0) + 301 (from 1,0) = 802
  // key 2: 201 (from 0,1)
  // key 3: 501 (from 1,1)
  EXPECT_THAT(val_out->AsSpan<int64_t>(), UnorderedElementsAre(802, 201, 501));
}

TEST_F(DpUnitCommitTest, SuccessDpUnitDeterminedByEventTime) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input 1 (privacy_id = 1):
  // +-----+-----+---------------------------+
  // | key | val | event_time                |
  // +-----+-----+---------------------------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00 |
  // +-----+-----+---------------------------+
  //
  // Input 2 (privacy_id = 1):
  // +-----+-----+---------------------------+
  // | key | val | event_time                |
  // +-----+-----+---------------------------+
  // |  1  | 200 | 2025-01-01T13:00:00+00:00 |
  // +-----+-----+---------------------------+
  DpUnitParameters dp_unit_parameters;
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;

  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1}, {100}, {"2025-01-01T12:00:00+00:00"}, {});
  ASSERT_THAT(tensors1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors2 =
      BuildTensors({1}, {200}, {"2025-01-01T13:00:00+00:00"}, {});
  ASSERT_THAT(tensors2, IsOk());

  // Same privacy ID for both inputs.
  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());
  absl::StatusOr<Tensor> privacy_id_tensor2 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor2, IsOk());
  absl::StatusOr<Input> input2 = Input::CreateFromTensors(
      *std::move(tensors2), {}, *std::move(privacy_id_tensor2));
  ASSERT_THAT(input2, IsOk());

  // With no DP columns and the same privacy ID, the data is split into two DP
  // units because the event times fall into different time windows.
  // DP Unit 1 (from input 1):
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 100 |
  // +-----+-----+
  // Output of SQL query: (1, 101)
  //
  // DP Unit 2 (from input 2):
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 200 |
  // +-----+-----+
  // Output of SQL query: (1, 201)

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input2)), IsOk());
  ASSERT_THAT(input_processor.CommitRowsGroupingByDpUnit(),
              IsOkAndHolds(IsEmpty()));

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      checkpoint_builder = builder_factory.Create();
  EXPECT_THAT(aggregator_->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());

  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_THAT(parser, IsOk());
  auto key_out = (*parser)->GetTensor("key_out");
  ASSERT_THAT(key_out, IsOk());
  EXPECT_THAT(key_out->AsSpan<int64_t>(), UnorderedElementsAre(1));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  // Final aggregated result:
  // +---------+---------+
  // | key_out | val_out |
  // +---------+---------+
  // |    1    |   302   |
  // +---------+---------+
  // We expect the SQL query to be run twice for key 1 since it's present in two
  // DP units. `(100 + 1) + (200 + 1) = 302`.
  EXPECT_THAT(val_out->AsSpan<int64_t>(), UnorderedElementsAre(302));
}

TEST_F(DpUnitCommitTest, PartialSqlError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input (privacy_id = 1):
  // +-----+-----+---------------------+
  // | key | val | event_time                |
  // +-----+-----+---------------------------+
  // |  1  |  1  | 2025-01-01T12:00:00+00:00 |
  // |  2  |  0  | 2025-01-02T13:00:00+00:00 |
  // +-----+-----+---------------------------+
  absl::StatusOr<std::vector<Tensor>> tensors1 = BuildTensors(
      {1, 2}, {1, 0},
      {"2025-01-01T12:00:00+00:00", "2025-01-02T13:00:00+00:00"}, {});
  ASSERT_THAT(tensors1, IsOk());

  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());
  sql_config_.query = "SELECT key, 1 / val AS val FROM input";
  DpUnitParameters dp_unit_parameters;
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  auto result = input_processor.CommitRowsGroupingByDpUnit();
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 1);
  EXPECT_THAT(result->at(0), StatusIs(absl::StatusCode::kInvalidArgument));

  // The aggregator should have received data from the successful DP unit.
  EXPECT_EQ(aggregator_->GetNumCheckpointsAggregated().value(), 1);

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      checkpoint_builder = builder_factory.Create();
  EXPECT_THAT(aggregator_->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());

  auto parser = parser_factory.Create(*checkpoint);
  ASSERT_THAT(parser, IsOk());
  auto key_out = (*parser)->GetTensor("key_out");
  ASSERT_THAT(key_out, IsOk());
  EXPECT_THAT(key_out->AsSpan<int64_t>(), ::testing::ElementsAreArray({1}));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  EXPECT_THAT(val_out->AsSpan<int64_t>(), ::testing::ElementsAreArray({1}));
}

TEST_F(DpUnitCommitTest, SqlError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input (privacy_id = 1):
  // +-----+-----+---------------------+-------+
  // | key | val | event_time                | unit1 |
  // +-----+-----+---------------------------+-------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00 |   0   |
  // |  2  | 200 | 2025-01-01T12:00:00+00:00 |   1   |
  // +-----+-----+---------------------------+-------+
  DpUnitParameters dp_unit_parameters;
  dp_unit_parameters.column_names.push_back("unit1");
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1, 2}, {100, 200},
                   {"2025-01-01T12:00:00+00:00", "2025-01-01T12:00:00+00:00"},
                   {{"unit1", {0, 1}}});
  ASSERT_THAT(tensors1, IsOk());

  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());

  // Invalid SQL query will cause an error for each DP unit.
  sql_config_.query = "SELECT invalid_column FROM input";

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  auto result = input_processor.CommitRowsGroupingByDpUnit();
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 2);
  EXPECT_THAT(result->at(0), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result->at(1), StatusIs(absl::StatusCode::kInvalidArgument));

  // The aggregator should not have received any data.
  EXPECT_EQ(aggregator_->GetNumCheckpointsAggregated().value(), 0);
}

TEST_F(DpUnitCommitTest, AccumulateError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input (privacy_id = 1):
  // +-----+-----+---------------------+-------+
  // | key | val | event_time                | unit1 |
  // +-----+-----+---------------------------+-------+
  // |  1  | 100 | 2025-01-01T12:00:00+00:00 |   0   |
  // +-----+-----+---------------------------+-------+
  DpUnitParameters dp_unit_parameters;
  dp_unit_parameters.column_names.push_back("unit1");
  *dp_unit_parameters.windowing_schedule.mutable_civil_time_window_schedule() =
      schedule_;
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      BuildTensors({1}, {100}, {"2025-01-01T12:00:00+00:00"}, {{"unit1", {0}}});
  ASSERT_THAT(tensors1, IsOk());

  absl::StatusOr<Tensor> privacy_id_tensor1 = CreatePrivacyIdTensor(id1_);
  ASSERT_THAT(privacy_id_tensor1, IsOk());
  absl::StatusOr<Input> input1 = Input::CreateFromTensors(
      *std::move(tensors1), {}, *std::move(privacy_id_tensor1));
  ASSERT_THAT(input1, IsOk());

  // Output column name "wrong_name" doesn't match aggregator expectation.
  sql_config_.output_columns.at(0).set_name("wrong_name");
  sql_config_.query = "SELECT key AS wrong_name, val FROM input";

  // Errors with the aggregator are propagated, since the aggregator may be in
  // an invalid state.
  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.StageInputForCommit(*std::move(input1)), IsOk());
  EXPECT_THAT(input_processor.CommitRowsGroupingByDpUnit(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(DpUnitCommitTest, ComputeDpTimeUnitHasNoWindowingSchedule) {
  DpUnitParameters dp_unit_parameters;  // Empty parameters.
  auto processor = DpUnitProcessor::Create(sql_config_, dp_unit_parameters,
                                           aggregator_.get());
  EXPECT_THAT(
      (*processor)->ComputeDPTimeUnit(absl::CivilSecond(2025, 1, 1, 0, 0, 0)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Windowing schedule must have civil time window schedule."));
}

class DpUnitHashTest : public Test {
 protected:
  void SetUp() override {
    CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());

    SqlConfiguration sql_config;  // Empty, not used by ComputeDPUnitHash.
    DpUnitParameters dp_unit_parameters;

    // Create a windowing schedule.
    auto schedule = PARSE_TEXT_PROTO(R"pb(
      size { size: 1 unit: HOURS }
      shift { size: 1 unit: HOURS }
      start_date { year: 2025 month: 1 day: 1 }
    )pb");
    *dp_unit_parameters.windowing_schedule
         .mutable_civil_time_window_schedule() = schedule;

    // The aggregator is not used by ComputeDPUnitHash.
    dp_unit_processor_ = std::make_unique<DpUnitProcessor>(
        sql_config, dp_unit_parameters, nullptr /* aggregator */);

    // Common data setup
    time1_ = absl::CivilSecond(2024, 1, 1, 0, 0, 0);
    time2_ = absl::CivilSecond(2024, 1, 1, 0, 0, 1);  // Different time
    id1_ = "1";
    id2_ = "2";  // Different ID

    std::unique_ptr<MutableVectorData<int64_t>> int64_data =
        std::make_unique<MutableVectorData<int64_t>>(0);
    int64_data->push_back(10);
    int64_data->push_back(20);
    tensors_.push_back(Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                                      std::move(int64_data), "col1")
                           .value());

    std::unique_ptr<MutableStringData> string_data =
        std::make_unique<MutableStringData>(2);
    string_data->Add("foo");
    string_data->Add("bar");
    tensors_.push_back(Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                                      std::move(string_data), "col2")
                           .value());

    row_view0_.emplace(RowView::CreateFromTensors(tensors_, 0).value());
    row_view1_.emplace(RowView::CreateFromTensors(tensors_, 1).value());
  }

  std::unique_ptr<DpUnitProcessor> dp_unit_processor_;
  absl::CivilSecond time1_;
  absl::CivilSecond time2_;
  std::string id1_;
  std::string id2_;
  std::vector<Tensor> tensors_;
  absl::optional<RowView> row_view0_;
  absl::optional<RowView> row_view1_;
};

TEST_F(DpUnitHashTest, ComputeDPUnitHashSameInputIsDeterministic) {
  // Hashing the same thing twice produces the same result.
  absl::StatusOr<uint64_t> hash1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash1, IsOk());

  absl::StatusOr<uint64_t> hash2 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash2, IsOk());

  EXPECT_EQ(*hash1, *hash2);
}

TEST_F(DpUnitHashTest, ComputeDPUnitHashDifferentRowsProducesDifferentHashes) {
  // Hashing column 0 for row 0 (data: 10)
  absl::StatusOr<uint64_t> hash_row0 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_row0, IsOk());

  // Hashing column 0 for row 1 (data: 20)
  absl::StatusOr<uint64_t> hash_row1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view1_, {0});
  ASSERT_THAT(hash_row1, IsOk());

  EXPECT_NE(*hash_row0, *hash_row1);
}

TEST_F(DpUnitHashTest,
       ComputeDPUnitHashDifferentColumnsProducesDifferentHashes) {
  // Hash for row 0, column 0 (data: 10)
  absl::StatusOr<uint64_t> hash_col0 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_col0, IsOk());

  // Hash for row 0, column 1 (data: "foo")
  absl::StatusOr<uint64_t> hash_col1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {1});
  ASSERT_THAT(hash_col1, IsOk());

  EXPECT_NE(*hash_col0, *hash_col1);
}

TEST_F(DpUnitHashTest,
       ComputeDPUnitHashDifferentColumnCombinationsProducesDifferentHashes) {
  absl::StatusOr<uint64_t> hash_col0 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_col0, IsOk());

  absl::StatusOr<uint64_t> hash_col1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {1});
  ASSERT_THAT(hash_col1, IsOk());

  // Hash for both columns
  absl::StatusOr<uint64_t> hash_both_cols =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0, 1});
  ASSERT_THAT(hash_both_cols, IsOk());

  EXPECT_NE(*hash_col0, *hash_both_cols);
  EXPECT_NE(*hash_col1, *hash_both_cols);
}

TEST_F(DpUnitHashTest,
       ComputeDPUnitHashDifferentColumnSelectionOrderProducesDifferentHashes) {
  // This test checks that the *order* of indices in the vector matters.
  // Hash with column order {0, 1}
  absl::StatusOr<uint64_t> hash_order_01 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0, 1});
  ASSERT_THAT(hash_order_01, IsOk());

  // Hash with column order {1, 0}
  absl::StatusOr<uint64_t> hash_order_10 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {1, 0});
  ASSERT_THAT(hash_order_10, IsOk());

  EXPECT_NE(*hash_order_01, *hash_order_10);
}

TEST_F(DpUnitHashTest, ComputeDPUnitHashDifferentIdsProducesDifferentHashes) {
  // Test sensitivity to the privacy ID.
  absl::StatusOr<uint64_t> hash_id1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_id1, IsOk());

  // Use id2_ instead of id1_
  absl::StatusOr<uint64_t> hash_id2 =
      dp_unit_processor_->ComputeDPUnitHash(id2_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_id2, IsOk());

  EXPECT_NE(*hash_id1, *hash_id2);
}

TEST_F(DpUnitHashTest,
       ComputeDPUnitHash_DifferentTimes_ProducesDifferentHashes) {
  // Test sensitivity to the time unit.
  absl::StatusOr<uint64_t> hash_time1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_time1, IsOk());

  // Use time2_ instead of time1_
  absl::StatusOr<uint64_t> hash_time2 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time2_, *row_view0_, {0});
  ASSERT_THAT(hash_time2, IsOk());

  EXPECT_NE(*hash_time1, *hash_time2);
}

TEST_F(DpUnitHashTest,
       ComputeDPUnitHashEmptyColumnListIsDeterministicAndUnique) {
  absl::StatusOr<uint64_t> hash_empty1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {});
  ASSERT_THAT(hash_empty1, IsOk());

  // Hash with an empty list again to ensure determinism
  absl::StatusOr<uint64_t> hash_empty2 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {});
  ASSERT_THAT(hash_empty2, IsOk());

  EXPECT_EQ(*hash_empty1, *hash_empty2);

  // Hash with a non-empty list to ensure the empty hash is unique
  absl::StatusOr<uint64_t> hash_col0 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash_col0, IsOk());

  EXPECT_NE(*hash_empty1, *hash_col0);
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
