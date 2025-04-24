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
#include "containers/fed_sql/kms_session.h"

#include <filesystem>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/fed_sql/testing/mocks.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::fed_sql::testing::
    BuildFedSqlGroupByCheckpoint;
using ::confidential_federated_compute::fed_sql::testing::MockInferenceModel;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Value;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Test;

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

KmsFedSqlSession CreateDefaultSession() {
  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  std::vector<Intrinsic> intrinsics =
      tensorflow_federated::aggregation::ParseFromConfig(DefaultConfiguration())
          .value();
  std::shared_ptr<InferenceModel> inference_model;
  return KmsFedSqlSession(std::move(checkpoint_aggregator), intrinsics,
                          inference_model, 99, 42, "sensitive_values_key");
}

TEST(KmsFedSqlSessionConfigureTest, ConfigureSessionSucceeds) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table { column { name: "col1" type: STRING } } }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.ConfigureSession(request), IsOk());
}

TEST(KmsFedSqlSessionConfigureTest, InvalidRequestFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  // Missing configure field.
  EXPECT_THAT(session.ConfigureSession(request),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, InvalidSqlQueryFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  // Incorrect configuration proto message type
  Value value;
  request.mutable_configure()->mutable_configuration()->PackFrom(value);

  EXPECT_THAT(session.ConfigureSession(request),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, InvalidDatabaseSchemaFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  // Missing database_schema
  SessionRequest request;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.ConfigureSession(request),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, ZeroInputColumnsFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  // Table schema missing columns
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table {} }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.ConfigureSession(request),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, AlreadyConfiguredFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table { column { name: "col1" type: STRING } } }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

  // First configuration should succeed.
  EXPECT_THAT(session.ConfigureSession(request), IsOk());
  // Second configuration should fail.
  EXPECT_THAT(session.ConfigureSession(request),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

class KmsFedSqlSessionWriteTest : public Test {
 public:
  KmsFedSqlSessionWriteTest() {
    std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
        CheckpointAggregator::Create(DefaultConfiguration()).value();
    intrinsics_ = tensorflow_federated::aggregation::ParseFromConfig(
                      DefaultConfiguration())
                      .value();
    session_ = std::make_unique<KmsFedSqlSession>(KmsFedSqlSession(
        std::move(checkpoint_aggregator), intrinsics_, mock_inference_model_,
        99, 42, "sensitive_values_key"));
    SessionRequest request;
    SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
      raw_sql: "SELECT key, val * 2 AS val FROM input"
      database_schema {
        table {
          name: "input"
          column { name: "key" type: INT64 }
          column { name: "val" type: INT64 }
          create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER)"
        }
      }
      output_columns { name: "key" type: INT64 }
      output_columns { name: "val" type: INT64 }
    )pb");
    request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

    CHECK_OK(session_->ConfigureSession(request));
  }

  ~KmsFedSqlSessionWriteTest() override {
    // Clean up any temp files created by the server.
    // for (auto& de : std::filesystem::directory_iterator("/tmp")) {
    // std::filesystem::remove_all(de.path());
    // }
  }

 protected:
  std::unique_ptr<KmsFedSqlSession> session_;
  std::vector<Intrinsic> intrinsics_;
  std::shared_ptr<NiceMock<MockInferenceModel>> mock_inference_model_ =
      std::make_shared<NiceMock<MockInferenceModel>>();
};

TEST_F(KmsFedSqlSessionWriteTest, InvalidWriteConfigurationFails) {
  WriteRequest write_request;
  Value value;
  write_request.mutable_first_request_configuration()->PackFrom(value);
  auto result = session_->SessionWrite(write_request, "unused");
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().write().status().code(),
              Eq(grpc::StatusCode::INVALID_ARGUMENT));
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateCommitSerializeSucceeds) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.mutable_first_request_metadata()->set_total_size_bytes(
      data.size());

  // Write the same unencrypted checkpoint twice.
  auto write_result_1 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result_2 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(data.size()));

  CommitRequest commit_request;
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_THAT(commit_response, IsOk());
  EXPECT_THAT(commit_response->commit().status().code(),
              Eq(grpc::StatusCode::OK));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  auto finalize_response =
      session_->FinalizeSession(finalize_request, metadata);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  ASSERT_GT(
      finalize_response->read().first_response_metadata().total_size_bytes(),
      0);
  ASSERT_TRUE(
      finalize_response->read().first_response_metadata().has_unencrypted());

  std::string result_data = finalize_response->read().data();
  ASSERT_THAT(UnbundlePrivateState(result_data), IsOk());
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_TRUE((*deserialized_agg)->Report(*checkpoint_builder).ok());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query doubles each element of the val column and sums them.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateCommitReportSucceeds) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.mutable_first_request_metadata()->set_total_size_bytes(
      data.size());

  // Write the same unencrypted checkpoint twice.
  auto write_result_1 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result_2 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(data.size()));

  CommitRequest commit_request;
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_THAT(commit_response, IsOk());
  EXPECT_THAT(commit_response->commit().status().code(),
              Eq(grpc::StatusCode::OK));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  auto finalize_response =
      session_->FinalizeSession(finalize_request, metadata);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  ASSERT_GT(
      finalize_response->read().first_response_metadata().total_size_bytes(),
      0);
  ASSERT_TRUE(
      finalize_response->read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response->read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles each input and the aggregation sums the
  // input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(KmsFedSqlSessionWriteTest,
       AccumulateSerializeWithoutCommitReturnsEmptyCheckpoint) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.mutable_first_request_metadata()->set_total_size_bytes(
      data.size());

  auto write_result = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result, IsOk());
  EXPECT_THAT(write_result.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result.value().write().committed_size_bytes(),
              Eq(data.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  auto finalize_response =
      session_->FinalizeSession(finalize_request, metadata);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  ASSERT_GT(
      finalize_response->read().first_response_metadata().total_size_bytes(),
      0);
  ASSERT_TRUE(
      finalize_response->read().first_response_metadata().has_unencrypted());

  std::string result_data = finalize_response->read().data();
  ASSERT_THAT(UnbundlePrivateState(result_data), IsOk());
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_TRUE((*deserialized_agg)->Report(*checkpoint_builder).ok());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  // The checkpoint contains the correct columns but they're empty since no
  // writes were committed.
  auto col_values = (*parser)->GetTensor("val_out");
  ASSERT_EQ(col_values->num_elements(), 0);
}

TEST_F(KmsFedSqlSessionWriteTest, MergeSerializeSucceeds) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_TRUE(input_aggregator->Accumulate(*input_parser).ok());

  std::string data = BundlePrivateState(
      (std::move(*input_aggregator).Serialize()).value(), PrivateState{});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.mutable_first_request_metadata()->set_total_size_bytes(
      data.size());

  write_request.mutable_first_request_metadata()->mutable_unencrypted();

  // Write the same unencrypted checkpoint twice.
  auto write_result_1 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result_2 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(data.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  auto finalize_response =
      session_->FinalizeSession(finalize_request, metadata);

  ASSERT_TRUE(finalize_response->read().finish_read());
  ASSERT_GT(
      finalize_response->read().first_response_metadata().total_size_bytes(),
      0);
  ASSERT_TRUE(
      finalize_response->read().first_response_metadata().has_unencrypted());

  std::string result_data = finalize_response->read().data();
  ASSERT_THAT(UnbundlePrivateState(result_data), IsOk());
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_TRUE((*deserialized_agg)->Report(*checkpoint_builder).ok());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The aggregation sums the val column.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

TEST_F(KmsFedSqlSessionWriteTest, MergeReportSucceeds) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_TRUE(input_aggregator->Accumulate(*input_parser).ok());

  std::string data = BundlePrivateState(
      (std::move(*input_aggregator).Serialize()).value(), PrivateState{});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.mutable_first_request_metadata()->set_total_size_bytes(
      data.size());

  write_request.mutable_first_request_metadata()->mutable_unencrypted();

  // Write the same unencrypted checkpoint twice.
  auto write_result_1 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result_2 = session_->SessionWrite(write_request, data);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(data.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  auto finalize_response =
      session_->FinalizeSession(finalize_request, metadata);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  ASSERT_GT(
      finalize_response->read().first_response_metadata().total_size_bytes(),
      0);
  ASSERT_TRUE(
      finalize_response->read().first_response_metadata().has_unencrypted());

  absl::Cord wire_format_result(finalize_response->read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
