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
#include <limits>
#include <memory>
#include <string>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/big_endian.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/range_tracker.h"
#include "containers/fed_sql/testing/mocks.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "containers/testing/mocks.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/struct.pb.h"
#include "google/rpc/code.pb.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fed_sql::testing::
    BuildFedSqlGroupByCheckpoint;
using ::confidential_federated_compute::fed_sql::testing::
    BuildMessageCheckpoint;
using ::confidential_federated_compute::fed_sql::testing::MessageHelper;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerPartitionedOutputConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalResultConfiguration;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::google::protobuf::Value;
using ::google::rpc::Code;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Pair;
using ::testing::Return;
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

std::shared_ptr<PrivateState> CreatePrivateState(
    const std::string& initial_state, std::optional<uint32_t> default_budget) {
  auto private_state =
      std::make_shared<PrivateState>(initial_state, default_budget);
  EXPECT_THAT(private_state->budget.Parse(initial_state), IsOk());
  return private_state;
}

KmsFedSqlSession CreateDefaultSession() {
  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  std::vector<Intrinsic> intrinsics =
      tensorflow_federated::aggregation::ParseFromConfig(DefaultConfiguration())
          .value();
  return KmsFedSqlSession(std::move(checkpoint_aggregator), intrinsics,
                          std::nullopt, std::nullopt, CreatePrivateState("", 1),
                          {},
                          /*prototype=*/nullptr, "test_query");
}

BlobMetadata MakeBlobMetadata(absl::string_view data, uint64_t blob_id,
                              absl::string_view key_id) {
  BlobHeader blob_header;
  *blob_header.mutable_blob_id() =
      StoreBigEndian(absl::MakeUint128(blob_id, 0));
  *blob_header.mutable_key_id() = std::string(key_id);

  BlobMetadata metadata;
  metadata.set_total_size_bytes(data.size());
  auto* hpke_plus_aead_data = metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_data->set_blob_id(blob_header.blob_id());
  auto* kms_associated_data =
      hpke_plus_aead_data->mutable_kms_symmetric_key_associated_data();
  *kms_associated_data->mutable_record_header() =
      blob_header.SerializeAsString();
  return metadata;
}

TEST(KmsFedSqlSessionConfigureTest, ConfigureSucceeds) {
  KmsFedSqlSession session = CreateDefaultSession();
  ConfigureRequest request;
  MockContext context;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table { column { name: "col1" type: STRING } } }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.Configure(request, context), IsOk());
}

TEST(KmsFedSqlSessionConfigureTest, InvalidRequestFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  ConfigureRequest request;
  MockContext context;
  // Missing configuration field.
  EXPECT_THAT(session.Configure(request, context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, InvalidSqlQueryFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  ConfigureRequest request;
  MockContext context;
  // Incorrect configuration proto message type
  Value value;
  request.mutable_configuration()->PackFrom(value);

  EXPECT_THAT(session.Configure(request, context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, InvalidDatabaseSchemaFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  // Missing database_schema
  ConfigureRequest request;
  MockContext context;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.Configure(request, context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, ZeroInputColumnsFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  ConfigureRequest request;
  MockContext context;
  // Table schema missing columns
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table {} }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configuration()->PackFrom(sql_query);

  EXPECT_THAT(session.Configure(request, context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KmsFedSqlSessionConfigureTest, AlreadyConfiguredFails) {
  KmsFedSqlSession session = CreateDefaultSession();
  ConfigureRequest request;
  MockContext context;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table { column { name: "col1" type: STRING } } }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configuration()->PackFrom(sql_query);

  // First configuration should succeed.
  EXPECT_THAT(session.Configure(request, context), IsOk());
  // Second configuration should fail.
  EXPECT_THAT(session.Configure(request, context),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

class KmsFedSqlSessionWriteTest : public Test {
 public:
  KmsFedSqlSessionWriteTest(std::optional<uint32_t> default_budget = 5) {
    intrinsics_ = tensorflow_federated::aggregation::ParseFromConfig(
                      DefaultConfiguration())
                      .value();
    std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
        CheckpointAggregator::Create(DefaultConfiguration()).value();
    BudgetState budget_state = PARSE_TEXT_PROTO(R"pb(
      buckets { key: "key_foo" budget: 1 }
      buckets { key: "key_bar" budget: 3 }
      buckets { key: "key_baz" budget: 0 }
    )pb");
    initial_private_state_ = budget_state.SerializeAsString();
    session_ = std::make_unique<KmsFedSqlSession>(
        std::move(checkpoint_aggregator), intrinsics_, std::nullopt,
        std::nullopt,
        CreatePrivateState(initial_private_state_, default_budget),
        absl::flat_hash_set<std::string>(),
        /*prototype=*/nullptr, "test_query");
    ConfigureRequest request;
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
    request.mutable_configuration()->PackFrom(sql_query);

    CHECK_OK(session_->Configure(request, context_));
  }

  ~KmsFedSqlSessionWriteTest() override {
    // Clean up any temp files created by the server.
    // for (auto& de : std::filesystem::directory_iterator("/tmp")) {
    // std::filesystem::remove_all(de.path());
    // }
  }

 protected:
  void ExpectEmitEncrypted(int& index, std::string& data) {
    EXPECT_CALL(context_, EmitEncrypted)
        .WillOnce([&](int reencryption_key_index, KmsFedSqlSession::KV kv) {
          index = reencryption_key_index;
          data = std::move(kv.data);
          return true;
        });
  }

  void ExpectEmitEncrypted(int& index, std::vector<std::string>& data,
                           std::vector<google::protobuf::Any>& configs) {
    EXPECT_CALL(context_, EmitEncrypted)
        .WillRepeatedly(
            [&](int reencryption_key_index, KmsFedSqlSession::KV kv) {
              index = reencryption_key_index;
              data.push_back(std::move(kv.data));
              configs.push_back(std::move(kv.key));
              return true;
            });
  }

  void ExpectEmitReleasable(int& index, std::optional<std::string>& src,
                            std::string& dst, std::string& data) {
    EXPECT_CALL(context_, EmitReleasable)
        .WillOnce([&](int reencryption_key_index, KmsFedSqlSession::KV kv,
                      std::optional<absl::string_view> src_state,
                      absl::string_view dst_state, std::string& release_token) {
          index = reencryption_key_index;
          src = src_state;
          dst = std::string(dst_state);
          data = std::move(kv.data);
          return true;
        });
  }

  std::unique_ptr<KmsFedSqlSession> session_;
  std::vector<Intrinsic> intrinsics_;
  std::string initial_private_state_;
  MockContext context_;
};

TEST_F(KmsFedSqlSessionWriteTest, InvalidWriteConfigurationFails) {
  WriteRequest write_request;
  Value value;
  write_request.mutable_first_request_configuration()->PackFrom(value);
  auto result = session_->Write(write_request, "unused", context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->status().code(), Code::INVALID_ARGUMENT)
      << result->status().message();
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateCommitSerializeSucceeds) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 2, "key_bar");

  auto write_result1 = session_->Write(write_request1, data, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(data.size()));
  auto write_result2 = session_->Write(write_request2, data, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->status().code(), Code::OK)
      << commit_response->status().message();
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::string result_data;
  int index;
  ExpectEmitEncrypted(index, result_data);

  ASSERT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  EXPECT_EQ(index, 0);
  auto range_tracker = UnbundleRangeTracker(result_data);
  ASSERT_THAT(*range_tracker,
              UnorderedElementsAre(
                  Pair("key_foo", ElementsAre(Interval<uint64_t>(1, 3))),
                  Pair("key_bar", ElementsAre(Interval<uint64_t>(1, 3)))));
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query doubles each element of the val column and sums them.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateCommitPartitionSucceeds) {
  // Create data with two different keys.
  std::string data1 = BuildFedSqlGroupByCheckpoint({8}, {2});
  std::string data2 = BuildFedSqlGroupByCheckpoint({1003}, {2});

  // Write inputs to the container.
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(data1, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(data2, 2, "key_bar");

  auto write_result1 = session_->Write(write_request1, data1, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(data1.size()));
  auto write_result2 = session_->Write(write_request2, data2, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(data2.size()));

  // Commit the inputs.
  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->status().code(), Code::OK)
      << commit_response->status().message();
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 2);

  // Partition the result into 2 partitions.
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_PARTITION
    num_partitions: 2
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::vector<std::string> result_data;
  std::vector<google::protobuf::Any> configs;
  int index;
  ExpectEmitEncrypted(index, result_data, configs);
  ASSERT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());
  EXPECT_EQ(result_data.size(), 2);
  EXPECT_EQ(index, 0);

  // Verify the partitoned results.
  for (int i = 0; i < result_data.size(); i++) {
    auto range_tracker = UnbundleRangeTracker(result_data[i]);
    ASSERT_THAT(*range_tracker,
                UnorderedElementsAre(
                    Pair("key_foo", ElementsAre(Interval<uint64_t>(1, 3))),
                    Pair("key_bar", ElementsAre(Interval<uint64_t>(1, 3)))));
    EXPECT_EQ(range_tracker->GetPartitionIndex(), std::optional<uint64_t>(i));
    FedSqlContainerPartitionedOutputConfiguration partition_config;
    ASSERT_TRUE(configs[i].UnpackTo(&partition_config));
    EXPECT_EQ(partition_config.partition_index(), i);
  }
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateCommitReportSucceeds) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 2, "key_foo");

  auto write_result1 = session_->Write(write_request1, data, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(data.size()));
  auto write_result2 = session_->Write(write_request2, data, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->status().code(), Code::OK)
      << commit_response->status().message();
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  std::string result_data;
  std::optional<std::string> src;
  std::string dst;
  int index;
  ExpectEmitReleasable(index, src, dst, result_data);
  BlobMetadata unused;
  auto finalize_response =
      session_->Finalize(finalize_request, unused, context_);
  ASSERT_THAT(finalize_response, IsOk());

  EXPECT_EQ(index, 1);
  EXPECT_EQ(src, initial_private_state_);
  BudgetState new_state;
  EXPECT_TRUE(new_state.ParseFromString(dst));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets {
                  key: "key_foo"
                  budget: 0
                  consumed_range_start: 1
                  consumed_range_end: 3
                }
                buckets { key: "key_bar" budget: 3 }
                buckets { key: "key_baz" budget: 0 }
              )pb"));

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  ASSERT_THAT(parser, IsOk());
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles each input and the aggregation sums the
  // input column
  EXPECT_EQ(col_values->num_elements(), 1);
  EXPECT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateOfBlobWithNoBudgetFails) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  // "key_baz" bucket has zero budget
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_baz");
  auto write_result = session_->Write(write_request, data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_THAT(write_result->status().code(), Code::FAILED_PRECONDITION);
  EXPECT_THAT(write_result->status().message(),
              HasSubstr("No budget remaining"));
}

TEST_F(KmsFedSqlSessionWriteTest, AccumulateOfRepeatedBlobFails) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  auto write_result = session_->Write(write_request, data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_EQ(write_result->status().code(), Code::OK)
      << write_result->status().message();

  // Submitting the same request again should fail due to duplicating blob ID.
  write_result = session_->Write(write_request, data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_THAT(write_result->status().code(), Code::FAILED_PRECONDITION);
  EXPECT_THAT(write_result->status().message(),
              HasSubstr("Blob rejected due to duplicate ID"));
}

TEST_F(KmsFedSqlSessionWriteTest,
       AccumulateSerializeWithoutCommitReturnsEmptyCheckpoint) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "foo");

  auto write_result = session_->Write(write_request, data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_EQ(write_result->status().code(), Code::OK)
      << write_result->status().message();
  EXPECT_THAT(write_result->committed_size_bytes(), Eq(data.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::string result_data;
  int index;
  ExpectEmitEncrypted(index, result_data);

  ASSERT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  EXPECT_EQ(index, 0);
  ASSERT_THAT(UnbundleRangeTracker(result_data), IsOk());
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
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
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  std::string data = std::move(*input_aggregator).Serialize().value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 3);
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 4, 6);
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "");

  auto write_result1 = session_->Write(write_request1, blob1, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(blob1.size()));
  auto write_result2 = session_->Write(write_request2, blob2, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(blob2.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::string result_data;
  int index;
  ExpectEmitEncrypted(index, result_data);

  EXPECT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  EXPECT_EQ(index, 0);
  auto result_range_tracker = UnbundleRangeTracker(result_data);
  ASSERT_THAT(*result_range_tracker,
              UnorderedElementsAre(
                  Pair("key_foo", ElementsAre(Interval<uint64_t>(1, 3),
                                              Interval<uint64_t>(4, 6)))));

  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
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
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  std::string data = (std::move(*input_aggregator).Serialize()).value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 2);
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 2, 3);
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "key_foo");

  auto write_result1 = session_->Write(write_request1, blob1, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(blob1.size()));
  auto write_result2 = session_->Write(write_request2, blob2, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(blob2.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);

  std::string result_data;
  std::optional<std::string> src;
  std::string dst;
  int index;
  ExpectEmitReleasable(index, src, dst, result_data);

  BlobMetadata unused;
  auto finalize_response =
      session_->Finalize(finalize_request, unused, context_);
  ASSERT_THAT(finalize_response, IsOk());

  EXPECT_EQ(index, 1);
  EXPECT_EQ(src, initial_private_state_);
  BudgetState new_state;
  EXPECT_TRUE(new_state.ParseFromString(dst));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets {
                  key: "key_foo"
                  budget: 0
                  consumed_range_start: 1
                  consumed_range_end: 3
                }
                buckets { key: "key_bar" budget: 3 }
                buckets { key: "key_baz" budget: 0 }
              )pb"));
  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  ASSERT_THAT(parser, IsOk());
  auto col_values = (*parser)->GetTensor("val_out");
  EXPECT_EQ(col_values->num_elements(), 1);
  EXPECT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

TEST_F(KmsFedSqlSessionWriteTest, MergeReportPartitionSucceeds) {
  // Create input data.
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());
  std::string data = (std::move(*input_aggregator).Serialize()).value();

  // Create 2 blobs with different ranges for partition index 0.
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 2);
  range_tracker1.SetPartitionIndex(0);
  range_tracker1.SetExpiredKeys({"key_bar"});
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 2, 3);
  range_tracker2.SetPartitionIndex(0);
  range_tracker2.SetExpiredKeys({"key_bar", "key_baz"});
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  // Write the blobs.
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "key_foo");
  auto write_result1 = session_->Write(write_request1, blob1, context_);
  ASSERT_THAT(write_result1, IsOk());
  auto write_result2 = session_->Write(write_request2, blob2, context_);
  ASSERT_THAT(write_result2, IsOk());

  // Finalize with REPORT_PARTITION.
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT_PARTITION
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);

  std::string result_data;
  std::optional<std::string> src;
  std::string dst;
  int index;
  ExpectEmitReleasable(index, src, dst, result_data);

  BlobMetadata unused;
  auto finalize_response =
      session_->Finalize(finalize_request, unused, context_);
  ASSERT_THAT(finalize_response, IsOk());

  // Verify the result.
  EXPECT_EQ(index, 1);
  EXPECT_EQ(src, std::nullopt);
  RangeTrackerState new_state;
  EXPECT_TRUE(new_state.ParseFromString(dst));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "key_foo" values: 1 values: 3 }
                partition_index: 0
                expired_keys: "key_bar"
                expired_keys: "key_baz"
              )pb"));
  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  ASSERT_THAT(parser, IsOk());
  auto col_values = (*parser)->GetTensor("val_out");
  EXPECT_EQ(col_values->num_elements(), 1);
  EXPECT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

class KmsFedSqlSessionUnlimitedBudgetTest : public KmsFedSqlSessionWriteTest {
 public:
  KmsFedSqlSessionUnlimitedBudgetTest()
      : KmsFedSqlSessionWriteTest(/*default_budget=*/std::nullopt) {}

 protected:
  void ExpectEmitUnencrypted(std::string& data) {
    EXPECT_CALL(context_, EmitUnencrypted)
        .WillOnce([&](KmsFedSqlSession::KV kv) {
          data = std::move(kv.data);
          return true;
        });
  }
};

TEST_F(KmsFedSqlSessionUnlimitedBudgetTest, MergeReportPartitionSucceeds) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  std::string data = (std::move(*input_aggregator).Serialize()).value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 2);
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 2, 3);
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "key_foo");

  auto write_result1 = session_->Write(write_request1, blob1, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(blob1.size()));
  auto write_result2 = session_->Write(write_request2, blob2, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(blob2.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT_PARTITION
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  std::string result_data;
  ExpectEmitUnencrypted(result_data);
  BlobMetadata unused;
  EXPECT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  auto parser = parser_factory.Create(absl::Cord(result_data));
  ASSERT_THAT(parser, IsOk());
  auto col_values = (*parser)->GetTensor("val_out");
  EXPECT_EQ(col_values->num_elements(), 1);
  EXPECT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

TEST_F(KmsFedSqlSessionWriteTest, CommitRangeConflict) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  EXPECT_THAT(session_->Write(write_request, data, context_), IsOk());

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  EXPECT_THAT(session_->Commit(commit_request, context_), IsOk());

  // Do another write and commit with the same range again - it should fail
  // because it uses the same range.
  EXPECT_THAT(session_->Write(write_request, data, context_), IsOk());
  EXPECT_THAT(session_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KmsFedSqlSessionWriteTest, CommitBlobsOutsideOfRange) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  EXPECT_THAT(session_->Write(write_request, data, context_), IsOk());

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 2 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  EXPECT_THAT(session_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KmsFedSqlSessionWriteTest, MergeRangeConflict) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  std::string data = std::move(*input_aggregator).Serialize().value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 5);
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 4, 6);
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "");

  // The second merge should fail due to the overlapping range.
  EXPECT_THAT(session_->Write(write_request1, blob1, context_), IsOk());
  EXPECT_THAT(session_->Write(write_request2, blob2, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KmsFedSqlSessionWriteTest, MergeReportBudgetExhausted) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  std::string data = (std::move(*input_aggregator).Serialize()).value();

  // Use two buckets "key_bar" and "key_baz" the second of which has
  // no budget.
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_bar", 1, 2);
  std::string blob1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_baz", 2, 3);
  std::string blob2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(blob1, 1, "key_bar");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(blob2, 2, "key_baz");

  EXPECT_THAT(session_->Write(write_request1, blob1, context_), IsOk());
  EXPECT_THAT(session_->Write(write_request2, blob2, context_), IsOk());

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  EXPECT_THAT(session_->Finalize(finalize_request, unused, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

class KmsFedSqlSessionWritePartialRangeTest : public Test {
 public:
  KmsFedSqlSessionWritePartialRangeTest() {
    intrinsics_ = tensorflow_federated::aggregation::ParseFromConfig(
                      DefaultConfiguration())
                      .value();
    checkpoint_aggregator_ =
        CheckpointAggregator::Create(DefaultConfiguration()).value();
  }

  ~KmsFedSqlSessionWritePartialRangeTest() override {
    // Clean up any temp files created by the server.
    // for (auto& de : std::filesystem::directory_iterator("/tmp")) {
    // std::filesystem::remove_all(de.path());
    // }
  }

 protected:
  void SetPrivateStateAndConfigure(BudgetState budget_state) {
    initial_private_state_ = budget_state.SerializeAsString();
    session_ = std::make_unique<KmsFedSqlSession>(
        std::move(checkpoint_aggregator_), intrinsics_, std::nullopt,
        std::nullopt, CreatePrivateState(initial_private_state_, 5),
        absl::flat_hash_set<std::string>(),
        /*prototype=*/nullptr, "test_query");
    ConfigureRequest request;
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
    request.mutable_configuration()->PackFrom(sql_query);

    CHECK_OK(session_->Configure(request, context_));
  }

  void ExpectEmitEncrypted(int& index, std::string& data) {
    EXPECT_CALL(context_, EmitEncrypted)
        .WillOnce([&](int reencryption_key_index, KmsFedSqlSession::KV kv) {
          index = reencryption_key_index;
          data = std::move(kv.data);
          return true;
        });
  }

  std::unique_ptr<KmsFedSqlSession> session_;
  std::vector<Intrinsic> intrinsics_;
  std::string initial_private_state_;
  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator_;
  MockContext context_;
};

TEST_F(KmsFedSqlSessionWritePartialRangeTest,
       NoOverlapAccumulateCommitSerializeSucceeds) {
  SetPrivateStateAndConfigure(PARSE_TEXT_PROTO(R"pb(
    buckets {
      key: "key_foo"
      budget: 5
      consumed_range_start: 1
      consumed_range_end: 10
    }
    buckets { key: "key_bar" budget: 5 }
  )pb"));
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 2, "key_bar");

  auto write_result1 = session_->Write(write_request1, data, context_);
  ASSERT_THAT(write_result1, IsOk());
  EXPECT_EQ(write_result1->status().code(), Code::OK)
      << write_result1->status().message();
  EXPECT_THAT(write_result1->committed_size_bytes(), Eq(data.size()));
  auto write_result2 = session_->Write(write_request2, data, context_);
  ASSERT_THAT(write_result2, IsOk());
  EXPECT_EQ(write_result2->status().code(), Code::OK)
      << write_result2->status().message();
  EXPECT_THAT(write_result2->committed_size_bytes(), Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->status().code(), Code::OK)
      << commit_response->status().message();
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::string result_data;
  int index;
  ExpectEmitEncrypted(index, result_data);

  ASSERT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  EXPECT_EQ(index, 0);
  auto range_tracker = UnbundleRangeTracker(result_data);
  ASSERT_THAT(*range_tracker,
              UnorderedElementsAre(
                  Pair("key_foo", ElementsAre(Interval<uint64_t>(1, 3))),
                  Pair("key_bar", ElementsAre(Interval<uint64_t>(1, 3)))));
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query doubles each element of the val column and sums them.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(KmsFedSqlSessionWritePartialRangeTest, AccumulateOverlappingBlobFails) {
  SetPrivateStateAndConfigure(PARSE_TEXT_PROTO(R"pb(
    buckets {
      key: "key_foo"
      budget: 0
      consumed_range_start: 5
      consumed_range_end: 10
    }
  )pb"));
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 8, "key_foo");
  auto write_result = session_->Write(write_request, data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_EQ(write_result->status().code(), Code::FAILED_PRECONDITION);
}

class TestMessageFactory : public MessageFactory {
 public:
  explicit TestMessageFactory(const google::protobuf::Message* prototype)
      : prototype_(prototype) {}

  std::unique_ptr<google::protobuf::Message> NewMessage() const override {
    return std::unique_ptr<google::protobuf::Message>(prototype_->New());
  }

 private:
  const google::protobuf::Message* prototype_;
};

class KmsFedSqlSessionWriteWithMessageTest : public Test {
 public:
  KmsFedSqlSessionWriteWithMessageTest() {
    intrinsics_ = tensorflow_federated::aggregation::ParseFromConfig(
                      DefaultConfiguration())
                      .value();
    std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
        CheckpointAggregator::Create(DefaultConfiguration()).value();
    BudgetState budget_state = PARSE_TEXT_PROTO(R"pb(
      buckets { key: "key_foo" budget: 1 }
      buckets { key: "key_bar" budget: 3 }
      buckets { key: "key_baz" budget: 0 }
    )pb");
    initial_private_state_ = budget_state.SerializeAsString();
    session_ = std::make_unique<KmsFedSqlSession>(
        std::move(checkpoint_aggregator), intrinsics_, std::nullopt,
        std::nullopt, CreatePrivateState(initial_private_state_, 5),
        absl::flat_hash_set<std::string>(),
        std::make_shared<TestMessageFactory>(message_helper_.prototype()),
        "test_query");
    ConfigureRequest request;
    // The input table will have columns "key" and "val" (from message) and
    // "confidential_compute_event_time" (system column). The aggregator expects
    // "key" and "val".
    SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
      raw_sql: "SELECT key, val * 2 AS val FROM input"
      database_schema {
        table {
          name: "input"
          column { name: "key" type: INT64 }
          column { name: "val" type: INT64 }
          column { name: "confidential_compute_event_time" type: STRING }
          create_table_sql: "CREATE TABLE input (key INTEGER, val INTEGER, "
                            "confidential_compute_event_time STRING)"
        }
      }
      output_columns { name: "key" type: INT64 }
      output_columns { name: "val" type: INT64 }
    )pb");
    request.mutable_configuration()->PackFrom(sql_query);
    CHECK_OK(session_->Configure(request, context_));
  }

 protected:
  void ExpectEmitEncrypted(int& index, std::string& data) {
    EXPECT_CALL(context_, EmitEncrypted)
        .WillOnce([&](int reencryption_key_index, KmsFedSqlSession::KV kv) {
          index = reencryption_key_index;
          data = std::move(kv.data);
          return true;
        });
  }

  void ExpectEmitReleasable(std::string& data) {
    EXPECT_CALL(context_, EmitReleasable)
        .WillOnce([&](int reencryption_key_index, KmsFedSqlSession::KV kv,
                      std::optional<absl::string_view> src_state,
                      absl::string_view dst_state, std::string& release_token) {
          data = std::move(kv.data);
          return true;
        });
  }

  MessageHelper message_helper_;
  std::unique_ptr<KmsFedSqlSession> session_;
  std::vector<Intrinsic> intrinsics_;
  std::string initial_private_state_;
  MockContext context_;
};

TEST_F(KmsFedSqlSessionWriteWithMessageTest,
       AccumulateCommitSerializeSucceeds) {
  // 1. Create checkpoint with serialized messages
  std::vector<std::string> serialized_messages;
  serialized_messages.push_back(
      message_helper_.CreateMessage(8, 1)->SerializeAsString());
  serialized_messages.push_back(
      message_helper_.CreateMessage(8, 2)->SerializeAsString());
  std::vector<std::string> event_times = {"2023-01-01T00:00:00Z",
                                          "2023-01-01T00:00:01Z"};

  absl::StatusOr<std::string> data = BuildMessageCheckpoint(
      std::move(serialized_messages), std::move(event_times), "test_query");
  ASSERT_THAT(data, IsOk());

  // 2. Write, Commit, Finalize, Verify
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(*data, 1, "key_foo");

  auto write_result = session_->Write(write_request, *data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_EQ(write_result->status().code(), Code::OK);

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 1);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  std::string result_data;
  int index;
  ExpectEmitEncrypted(index, result_data);

  ASSERT_THAT(session_->Finalize(finalize_request, unused, context_), IsOk());

  EXPECT_EQ(index, 0);
  ASSERT_THAT(UnbundleRangeTracker(result_data), IsOk());
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query doubles each element of the val column and sums them.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

TEST_F(KmsFedSqlSessionWriteWithMessageTest, AccumulateCommitReportSucceeds) {
  // 1. Create checkpoint with serialized messages
  std::vector<std::string> serialized_messages;
  serialized_messages.push_back(
      message_helper_.CreateMessage(8, 1)->SerializeAsString());
  serialized_messages.push_back(
      message_helper_.CreateMessage(8, 2)->SerializeAsString());
  std::vector<std::string> event_times = {"2023-01-01T00:00:00Z",
                                          "2023-01-01T00:00:01Z"};

  absl::StatusOr<std::string> data = BuildMessageCheckpoint(
      std::move(serialized_messages), std::move(event_times), "test_query");
  ASSERT_THAT(data, IsOk());

  // 2. Write, Commit
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(*data, 1, "key_foo");

  auto write_result = session_->Write(write_request, *data, context_);
  ASSERT_THAT(write_result, IsOk());
  EXPECT_EQ(write_result->status().code(), Code::OK);

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->Commit(commit_request, context_);
  ASSERT_THAT(commit_response, IsOk());
  EXPECT_EQ(commit_response->stats().num_inputs_committed(), 1);

  // 3. Finalize and Verify
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  std::string result_data;
  ExpectEmitReleasable(result_data);
  BlobMetadata unused;
  auto finalize_response =
      session_->Finalize(finalize_request, unused, context_);
  ASSERT_THAT(finalize_response, IsOk());

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  ASSERT_THAT(parser, IsOk());
  auto col_values = (*parser)->GetTensor("val_out");
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 6);
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
