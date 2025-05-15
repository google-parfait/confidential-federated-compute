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
#include "containers/crypto_test_utils.h"
#include "containers/fed_sql/range_tracker.h"
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
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
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
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Pair;
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

KmsFedSqlSession CreateDefaultSession() {
  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  std::vector<Intrinsic> intrinsics =
      tensorflow_federated::aggregation::ParseFromConfig(DefaultConfiguration())
          .value();
  std::shared_ptr<InferenceModel> inference_model;
  auto merge_public_private_key_pair =
      crypto_test_utils::GenerateKeyPair("merge");
  auto report_public_private_key_pair =
      crypto_test_utils::GenerateKeyPair("report");
  return KmsFedSqlSession(
      std::move(checkpoint_aggregator), intrinsics, inference_model, 42,
      "sensitive_values_key",
      std::vector<std::string>{merge_public_private_key_pair.first,
                               report_public_private_key_pair.first});
}

void StoreBigEndian(void* p, uint64_t num) {
  // Assume that the current system is little endian
  // and swap bytes individually to convert to big endian.
  uint8_t* b = reinterpret_cast<uint8_t*>(&num);
  std::swap(b[0], b[7]);
  std::swap(b[1], b[6]);
  std::swap(b[2], b[5]);
  std::swap(b[3], b[4]);
  memcpy(p, &num, sizeof(uint64_t));
}

BlobMetadata MakeBlobMetadata(absl::string_view data, uint64_t blob_id,
                              absl::string_view key_id) {
  std::string binary_blob_id(16, '\0');
  StoreBigEndian(binary_blob_id.data(), blob_id);

  BlobHeader blob_header;
  *blob_header.mutable_blob_id() = std::move(binary_blob_id);
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
    auto merge_public_private_key_pair =
        crypto_test_utils::GenerateKeyPair("merge");
    auto report_public_private_key_pair =
        crypto_test_utils::GenerateKeyPair("report");
    google::protobuf::Struct config_properties;
    message_decryptor_ = std::make_unique<MessageDecryptor>(
        config_properties, std::vector<absl::string_view>(
                               {merge_public_private_key_pair.second,
                                report_public_private_key_pair.second}));
    intrinsics_ = tensorflow_federated::aggregation::ParseFromConfig(
                      DefaultConfiguration())
                      .value();
    std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
        CheckpointAggregator::Create(DefaultConfiguration()).value();
    session_ = std::make_unique<KmsFedSqlSession>(KmsFedSqlSession(
        std::move(checkpoint_aggregator), intrinsics_, mock_inference_model_,
        42, "sensitive_values_key",
        std::vector<std::string>{merge_public_private_key_pair.first,
                                 report_public_private_key_pair.first}));
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
  std::string Decrypt(BlobMetadata metadata, absl::string_view ciphertext) {
    BlobHeader blob_header;
    blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                    .kms_symmetric_key_associated_data()
                                    .record_header());
    auto decrypted = message_decryptor_->Decrypt(
        ciphertext, metadata.hpke_plus_aead_data().ciphertext_associated_data(),
        metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
        metadata.hpke_plus_aead_data()
            .kms_symmetric_key_associated_data()
            .record_header(),
        metadata.hpke_plus_aead_data().encapsulated_public_key(),
        blob_header.key_id());
    CHECK_OK(decrypted.status());
    return decrypted.value();
  }

  std::unique_ptr<KmsFedSqlSession> session_;
  std::vector<Intrinsic> intrinsics_;
  std::shared_ptr<NiceMock<MockInferenceModel>> mock_inference_model_ =
      std::make_shared<NiceMock<MockInferenceModel>>();
  std::unique_ptr<MessageDecryptor> message_decryptor_;
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

  auto write_result_1 = session_->SessionWrite(write_request1, data);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result_2 = session_->SessionWrite(write_request2, data);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_THAT(commit_response, IsOk());
  EXPECT_THAT(commit_response->commit().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_EQ(commit_response->commit().stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);
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
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "foo");

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
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 2 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_THAT(commit_response, IsOk());
  EXPECT_THAT(commit_response->commit().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_EQ(commit_response->commit().stats().num_inputs_committed(), 2);

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
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "foo");

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
  BlobMetadata unused;
  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_THAT(finalize_response, IsOk());
  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);

  ASSERT_THAT(UnbundleRangeTracker(result_data), IsOk());
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

  std::string data = std::move(*input_aggregator).Serialize().value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 3);
  std::string bundle1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 4, 6);
  std::string bundle2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(bundle1, 1, "");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(bundle2, 2, "");

  auto write_result_1 = session_->SessionWrite(write_request1, bundle1);
  EXPECT_THAT(write_result_1, IsOk());
  EXPECT_THAT(write_result_1.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_1.value().write().committed_size_bytes(),
              Eq(bundle1.size()));
  auto write_result_2 = session_->SessionWrite(write_request2, bundle2);
  EXPECT_THAT(write_result_2, IsOk());
  EXPECT_THAT(write_result_2.value().write().status().code(),
              Eq(grpc::StatusCode::OK));
  EXPECT_THAT(write_result_2.value().write().committed_size_bytes(),
              Eq(bundle2.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);

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

  std::string data = BundleRangeTracker(
      (std::move(*input_aggregator).Serialize()).value(), RangeTracker{});
  WriteRequest write_request;
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "foo");

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

TEST_F(KmsFedSqlSessionWriteTest, CommitRangeConflict) {
  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  WriteRequest write_request;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  *write_request.mutable_first_request_metadata() =
      MakeBlobMetadata(data, 1, "key_foo");
  EXPECT_THAT(session_->SessionWrite(write_request, data), IsOk());

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  EXPECT_THAT(session_->SessionCommit(commit_request), IsOk());

  // Do another write and commit with the same range again - it should fail
  // because it uses the same range.
  EXPECT_THAT(session_->SessionWrite(write_request, data), IsOk());
  EXPECT_THAT(session_->SessionCommit(commit_request),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KmsFedSqlSessionWriteTest, MergeRangeConflict) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_TRUE(input_aggregator->Accumulate(*input_parser).ok());

  std::string data = std::move(*input_aggregator).Serialize().value();
  RangeTracker range_tracker1;
  range_tracker1.AddRange("key_foo", 1, 5);
  std::string bundle1 = BundleRangeTracker(data, range_tracker1);
  RangeTracker range_tracker2;
  range_tracker2.AddRange("key_foo", 4, 6);
  std::string bundle2 = BundleRangeTracker(data, range_tracker2);

  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_MERGE
  )pb");
  WriteRequest write_request1;
  write_request1.mutable_first_request_configuration()->PackFrom(config);
  *write_request1.mutable_first_request_metadata() =
      MakeBlobMetadata(bundle1, 1, "");
  WriteRequest write_request2;
  write_request2.mutable_first_request_configuration()->PackFrom(config);
  *write_request2.mutable_first_request_metadata() =
      MakeBlobMetadata(bundle2, 2, "");

  // The second merge should fail due to the overlapping range.
  EXPECT_THAT(session_->SessionWrite(write_request1, bundle1), IsOk());
  EXPECT_THAT(session_->SessionWrite(write_request2, bundle2),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
