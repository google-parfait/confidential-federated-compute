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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/optional.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "containers/sql/input.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::RepeatedPtrField;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::Configuration;
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

class DpUnitTest : public ::testing::Test {
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

    dp_unit_processor_ = std::make_unique<DpUnitProcessor>(
        sql_config_, dp_unit_parameters_, aggregator_.get());

    // Common data setup
    time1_ = absl::CivilSecond(2024, 1, 1, 0, 0, 0);
    time2_ = absl::CivilSecond(2024, 1, 1, 0, 0, 1);  // Different time
    id1_ = 1;
    id2_ = 2;  // Different ID

    // Create tensors with two rows:
    // Row 0: {10, "foo"}
    // Row 1: {20, "bar"}
    std::unique_ptr<MutableVectorData<int64_t>> int64_data =
        std::make_unique<MutableVectorData<int64_t>>(0);
    int64_data->push_back(10);
    int64_data->push_back(20);
    tensors_array_[0] = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                                       std::move(int64_data), "col1")
                            .value();

    std::unique_ptr<MutableStringData> string_data =
        std::make_unique<MutableStringData>(2);
    string_data->Add("foo");
    string_data->Add("bar");
    tensors_array_[1] = Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                                       std::move(string_data), "col2")
                            .value();
    dp_tensors_ = absl::Span<const Tensor>(tensors_array_);

    // Use emplace to construct the RowView inside the optional
    row_view0_.emplace(
        confidential_federated_compute::sql::RowView::CreateFromTensors(
            dp_tensors_, 0)
            .value());
    row_view1_.emplace(
        confidential_federated_compute::sql::RowView::CreateFromTensors(
            dp_tensors_, 1)
            .value());
  }

  // Member variables.
  std::unique_ptr<CheckpointAggregator> aggregator_;
  SqlConfiguration sql_config_;
  fcp::confidentialcompute::WindowingSchedule::CivilTimeWindowSchedule
      schedule_;

  // Common test data.
  DpUnitParameters dp_unit_parameters_ = {.column_names = {"key"}};
  std::unique_ptr<DpUnitProcessor> dp_unit_processor_;
  absl::CivilSecond time1_;
  absl::CivilSecond time2_;
  int64_t id1_;
  int64_t id2_;
  Tensor tensors_array_[2];
  absl::Span<const Tensor> dp_tensors_;
  absl::optional<confidential_federated_compute::sql::RowView> row_view0_;
  absl::optional<confidential_federated_compute::sql::RowView> row_view1_;
};

TEST_F(DpUnitTest, Success) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input checkpoint 1 contains:
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 100 |
  // |  2  | 200 |
  // |  1  | 300 |
  // +-----+-----+
  //
  // Input checkpoint 2 contains:
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  | 400 |
  // |  3  | 500 |
  // +-----+-----+
  std::string checkpoint1 =
      testing::BuildFedSqlGroupByCheckpoint({1, 2, 1}, {100, 200, 300});
  auto parser1 = parser_factory.Create(absl::Cord(checkpoint1));
  ASSERT_THAT(parser1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      Deserialize(sql_config_.input_schema, parser1->get());
  ASSERT_THAT(tensors1, IsOk());
  std::string checkpoint2 =
      testing::BuildFedSqlGroupByCheckpoint({1, 3}, {400, 500});
  auto parser2 = parser_factory.Create(absl::Cord(checkpoint2));
  ASSERT_THAT(parser2, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors2 =
      Deserialize(sql_config_.input_schema, parser2->get());
  ASSERT_THAT(tensors2, IsOk());

  std::vector<Input> uncommitted_inputs;
  absl::StatusOr<Input> input1 =
      Input::CreateFromTensors(*std::move(tensors1), {});
  ASSERT_THAT(input1, IsOk());
  uncommitted_inputs.push_back(*std::move(input1));
  absl::StatusOr<Input> input2 =
      Input::CreateFromTensors(*std::move(tensors2), {});
  ASSERT_THAT(input2, IsOk());
  uncommitted_inputs.push_back(*std::move(input2));

  // For each DP unit, we run the SQL query `SELECT key, SUM(val) + 1 AS val
  // FROM input GROUP BY key` on its subset of rows. DP Unit 1 corresponds to:
  // +-----+-----+-------------+
  // | key | val | checkpoint  |
  // +-----+-----+-------------+
  // |  1  | 100 | checkpoint1 |
  // +-----+-----+-------------+
  // Output of SQL query: (1, 101)
  //
  // DP Unit 2 contains:
  // +-----+-----+-------------+
  // | key | val | checkpoint  |
  // +-----+-----+-------------+
  // |  2  | 200 | checkpoint1 |
  // |  1  | 400 | checkpoint2 |
  // |  1  | 300 | checkpoint1 |
  // +-----+-----+-------------+
  // Output of SQL query: (1, 701), (2, 201)
  //
  // DP Unit 3 processes:
  // +-----+-----+-------------+
  // | key | val | checkpoint  |
  // +-----+-----+-------------+
  // |  3  | 500 | checkpoint2 |
  // +-----+-----+-------------+
  // Output of SQL query: (3, 401)

  std::vector<RowLocation> row_dp_unit_index;
  // Insert the rows in a random order to ensure the code under test sorts them
  // correctly.
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 3, .input_index = 1, .row_index = 1});
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 2, .input_index = 0, .row_index = 1});
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 1, .input_index = 0, .row_index = 0});
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 2, .input_index = 1, .row_index = 0});
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 2, .input_index = 0, .row_index = 2});

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters_,
                                  aggregator_.get());
  ASSERT_THAT(input_processor.CommitRowsGroupingByDpUnit(
                  std::move(uncommitted_inputs), std::move(row_dp_unit_index)),
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
  EXPECT_THAT(key_out->AsSpan<int64_t>(),
              ::testing::ElementsAreArray({1, 2, 3}));
  auto val_out = (*parser)->GetTensor("val_out");
  ASSERT_THAT(val_out, IsOk());
  // Final aggregated result:
  // +---------+---------+
  // | key_out | val_out |
  // +---------+---------+
  // |    1    |   802   |
  // |    2    |   201   |
  // |    3    |   401   |
  // +---------+---------+
  // We expect the SQL query to be run twice for key 1 since it's present in two
  // DP units. The SQL query adds 1 each time it's executed.
  EXPECT_THAT(val_out->AsSpan<int64_t>(),
              ::testing::ElementsAreArray({802, 201, 501}));
}

TEST_F(DpUnitTest, PartialSqlError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  // Input checkpoint contains:
  // +-----+-----+
  // | key | val |
  // +-----+-----+
  // |  1  |  1 |
  // |  2  |  0 |
  // +-----+-----+
  std::string checkpoint1 =
      testing::BuildFedSqlGroupByCheckpoint({1, 2}, {1, 0});
  auto parser1 = parser_factory.Create(absl::Cord(checkpoint1));
  ASSERT_THAT(parser1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      Deserialize(sql_config_.input_schema, parser1->get());
  ASSERT_THAT(tensors1, IsOk());

  std::vector<Input> uncommitted_inputs;
  absl::StatusOr<Input> input1 =
      Input::CreateFromTensors(*std::move(tensors1), {});
  ASSERT_THAT(input1, IsOk());
  uncommitted_inputs.push_back(*std::move(input1));
  // Query will cause division by zero for DP unit 2.
  sql_config_.query = "SELECT key, 1 / val AS val FROM input";

  std::vector<RowLocation> row_dp_unit_index;
  // DP Unit 1 has row (1, 1).
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 1, .input_index = 0, .row_index = 0});
  // DP Unit 2 has rows (2, 0). This will cause a division by zero error when
  // executing the SQL query.
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 2, .input_index = 0, .row_index = 1});

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters_,
                                  aggregator_.get());
  auto result = input_processor.CommitRowsGroupingByDpUnit(
      std::move(uncommitted_inputs), std::move(row_dp_unit_index));
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

TEST_F(DpUnitTest, SqlError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  std::string checkpoint1 =
      testing::BuildFedSqlGroupByCheckpoint({1, 2}, {100, 200});
  auto parser1 = parser_factory.Create(absl::Cord(checkpoint1));
  ASSERT_THAT(parser1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      Deserialize(sql_config_.input_schema, parser1->get());
  ASSERT_THAT(tensors1, IsOk());

  std::vector<Input> uncommitted_inputs;
  absl::StatusOr<Input> input1 =
      Input::CreateFromTensors(*std::move(tensors1), {});
  ASSERT_THAT(input1, IsOk());
  uncommitted_inputs.push_back(*std::move(input1));

  std::vector<RowLocation> row_dp_unit_index;
  // DP Unit 1 has row (1, 100).
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 1, .input_index = 0, .row_index = 0});
  // DP Unit 2 has row (2, 200).
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 2, .input_index = 0, .row_index = 1});

  // Invalid SQL query will cause an error for each DP unit.
  sql_config_.query = "SELECT invalid_column FROM input";

  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters_,
                                  aggregator_.get());
  auto result = input_processor.CommitRowsGroupingByDpUnit(
      std::move(uncommitted_inputs), std::move(row_dp_unit_index));
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 2);
  EXPECT_THAT(result->at(0), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(result->at(1), StatusIs(absl::StatusCode::kInvalidArgument));

  // The aggregator should not have received any data.
  EXPECT_EQ(aggregator_->GetNumCheckpointsAggregated().value(), 0);
}

TEST_F(DpUnitTest, AccumulateError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  std::string checkpoint1 = testing::BuildFedSqlGroupByCheckpoint({1}, {100});
  auto parser1 = parser_factory.Create(absl::Cord(checkpoint1));
  ASSERT_THAT(parser1, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors1 =
      Deserialize(sql_config_.input_schema, parser1->get());
  ASSERT_THAT(tensors1, IsOk());

  std::vector<Input> uncommitted_inputs;
  absl::StatusOr<Input> input1 =
      Input::CreateFromTensors(*std::move(tensors1), {});
  ASSERT_THAT(input1, IsOk());
  uncommitted_inputs.push_back(*std::move(input1));

  std::vector<RowLocation> row_dp_unit_index;
  row_dp_unit_index.push_back(
      {.dp_unit_hash = 1, .input_index = 0, .row_index = 0});

  // Output column name "wrong_name" doesn't match aggregator expectation.
  sql_config_.output_columns.at(0).set_name("wrong_name");
  sql_config_.query = "SELECT key AS wrong_name, val FROM input";

  // Errors with the aggregator are propagated, since the aggregator may be in
  // an invalid state.
  DpUnitProcessor input_processor(sql_config_, dp_unit_parameters_,
                                  aggregator_.get());
  EXPECT_THAT(input_processor.CommitRowsGroupingByDpUnit(
                  std::move(uncommitted_inputs), std::move(row_dp_unit_index)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(DpUnitTest, ComputeDpTimeUnitHasNoWindowingSchedule) {
  DpUnitParameters dp_unit_parameters;  // Empty parameters.
  auto processor = DpUnitProcessor::Create(sql_config_, dp_unit_parameters,
                                           aggregator_.get());
  EXPECT_THAT(
      (*processor)->ComputeDPTimeUnit(absl::CivilSecond(2025, 1, 1, 0, 0, 0)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Windowing schedule must have civil time window schedule."));
}

TEST_F(DpUnitTest, ComputeDPUnitHashSameInputIsDeterministic) {
  // Hashing the same thing twice produces the same result.
  absl::StatusOr<uint64_t> hash1 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash1, IsOk());

  absl::StatusOr<uint64_t> hash2 =
      dp_unit_processor_->ComputeDPUnitHash(id1_, time1_, *row_view0_, {0});
  ASSERT_THAT(hash2, IsOk());

  EXPECT_EQ(*hash1, *hash2);
}

TEST_F(DpUnitTest, ComputeDPUnitHashDifferentRowsProducesDifferentHashes) {
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

TEST_F(DpUnitTest, ComputeDPUnitHashDifferentColumnsProducesDifferentHashes) {
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

TEST_F(DpUnitTest,
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

TEST_F(DpUnitTest,
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

TEST_F(DpUnitTest, ComputeDPUnitHashDifferentIdsProducesDifferentHashes) {
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

TEST_F(DpUnitTest, ComputeDPUnitHash_DifferentTimes_ProducesDifferentHashes) {
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

TEST_F(DpUnitTest, ComputeDPUnitHashEmptyColumnListIsDeterministicAndUnique) {
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
