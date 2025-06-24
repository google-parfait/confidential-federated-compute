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

#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "containers/big_endian.h"
#include "containers/crypto_test_utils.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/range_tracker.h"
#include "containers/fed_sql/testing/mocks.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
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

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::confidential_federated_compute::fed_sql::testing::
    BuildFedSqlGroupByCheckpoint;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::ReleaseToken;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalResultConfiguration;
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
    const std::string& initial_state, uint32_t default_budget) {
  auto private_state =
      std::make_shared<PrivateState>(initial_state, default_budget);
  EXPECT_OK(private_state->budget.Parse(initial_state));
  return private_state;
}

KmsFedSqlSession CreateDefaultSession() {
  std::unique_ptr<CheckpointAggregator> checkpoint_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  std::vector<Intrinsic> intrinsics =
      tensorflow_federated::aggregation::ParseFromConfig(DefaultConfiguration())
          .value();
  auto merge_public_private_key_pair =
      crypto_test_utils::GenerateKeyPair("merge");
  auto report_public_private_key_pair =
      crypto_test_utils::GenerateKeyPair("report");
  std::shared_ptr<MockSigningKeyHandle> mock_signing_key_handle =
      std::make_shared<MockSigningKeyHandle>();
  return KmsFedSqlSession(
      std::move(checkpoint_aggregator), intrinsics,
      SessionInferenceConfiguration(), "sensitive_values_key",
      std::vector<std::string>{merge_public_private_key_pair.first,
                               report_public_private_key_pair.first},
      "reencryption_policy_hash", CreatePrivateState("", 1),
      mock_signing_key_handle);
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

TEST(KmsFedSqlSessionConfigureTest, ConfigureSessionSucceeds) {
  KmsFedSqlSession session = CreateDefaultSession();
  SessionRequest request;
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT * FROM my_table"
    database_schema { table { column { name: "col1" type: STRING } } }
    output_columns { name: "col1" type: STRING }
  )pb");
  request.mutable_configure()->mutable_configuration()->PackFrom(sql_query);

  EXPECT_OK(session.ConfigureSession(request));
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
  EXPECT_OK(session.ConfigureSession(request));
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
    BudgetState budget_state = PARSE_TEXT_PROTO(R"pb(
      buckets { key: "key_foo" budget: 1 }
      buckets { key: "key_bar" budget: 3 }
      buckets { key: "key_baz" budget: 0 }
    )pb");
    initial_private_state_ = budget_state.SerializeAsString();
    session_ = std::make_unique<KmsFedSqlSession>(
        std::move(checkpoint_aggregator), intrinsics_,
        SessionInferenceConfiguration(), "sensitive_values_key",
        std::vector<std::string>{merge_public_private_key_pair.first,
                                 report_public_private_key_pair.first},
        "reencryption_policy_hash",
        CreatePrivateState(initial_private_state_, 5),

        mock_signing_key_handle_);
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
    CHECK(blob_header.access_policy_sha256() == "reencryption_policy_hash");
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
  std::unique_ptr<MessageDecryptor> message_decryptor_;
  std::shared_ptr<MockSigningKeyHandle> mock_signing_key_handle_ =
      std::make_shared<MockSigningKeyHandle>();
  std::string initial_private_state_;
};

TEST_F(KmsFedSqlSessionWriteTest, InvalidWriteConfigurationFails) {
  WriteRequest write_request;
  Value value;
  write_request.mutable_first_request_configuration()->PackFrom(value);
  auto result = session_->SessionWrite(write_request, "unused");
  EXPECT_OK(result);
  EXPECT_THAT(result.value().write().status(),
              IsCode(grpc::StatusCode::INVALID_ARGUMENT));
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

  auto write_result1 = session_->SessionWrite(write_request1, data);
  EXPECT_OK(write_result1);
  EXPECT_OK(write_result1.value().write().status());
  EXPECT_THAT(write_result1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result2 = session_->SessionWrite(write_request2, data);
  EXPECT_OK(write_result2);
  EXPECT_OK(write_result2.value().write().status());
  EXPECT_THAT(write_result2.value().write().committed_size_bytes(),
              Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_OK(commit_response);
  EXPECT_OK(commit_response->commit().status());
  EXPECT_EQ(commit_response->commit().stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_OK(finalize_response);
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
  ASSERT_OK(deserialized_agg);

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_OK((*deserialized_agg)->Report(*checkpoint_builder));
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

  auto write_result1 = session_->SessionWrite(write_request1, data);
  EXPECT_OK(write_result1);
  EXPECT_OK(write_result1.value().write().status());
  EXPECT_THAT(write_result1.value().write().committed_size_bytes(),
              Eq(data.size()));
  auto write_result2 = session_->SessionWrite(write_request2, data);
  EXPECT_OK(write_result2);
  EXPECT_OK(write_result2.value().write().status());
  EXPECT_THAT(write_result2.value().write().committed_size_bytes(),
              Eq(data.size()));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  auto commit_response = session_->SessionCommit(commit_request);
  EXPECT_OK(commit_response);
  EXPECT_OK(commit_response->commit().status());
  EXPECT_EQ(commit_response->commit().stats().num_inputs_committed(), 2);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  oak::crypto::v1::Signature fake_signature;
  fake_signature.set_signature("signature");
  EXPECT_CALL(*mock_signing_key_handle_, Sign(_))
      .WillOnce(Return(fake_signature));

  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_OK(finalize_response);
  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  // Verify the release token.
  FinalResultConfiguration final_result_config;
  ASSERT_TRUE(finalize_response->read().first_response_configuration().UnpackTo(
      &final_result_config));
  absl::StatusOr<ReleaseToken> release_token =
      ReleaseToken::Decode(final_result_config.release_token());
  ASSERT_OK(release_token);
  EXPECT_EQ(release_token->encryption_key_id, "report");
  EXPECT_EQ(release_token->src_state, initial_private_state_);
  EXPECT_EQ(release_token->signature, "signature");
  BudgetState new_state;
  EXPECT_TRUE(new_state.ParseFromString(release_token->dst_state.value()));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "key_foo" budget: 0 }
                buckets { key: "key_bar" budget: 3 }
                buckets { key: "key_baz" budget: 0 }
              )pb"));

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);

  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  EXPECT_OK(parser);
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
  auto write_result = session_->SessionWrite(write_request, data);
  EXPECT_OK(write_result);
  EXPECT_THAT(write_result.value().write().status(),
              IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(write_result.value().write().status().message(),
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
  auto write_result = session_->SessionWrite(write_request, data);
  EXPECT_OK(write_result);
  EXPECT_OK(write_result.value().write().status());

  // Submitting the same request again should fail due to duplicating blob ID.
  write_result = session_->SessionWrite(write_request, data);
  EXPECT_OK(write_result);
  EXPECT_THAT(write_result.value().write().status(),
              IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(write_result.value().write().status().message(),
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

  auto write_result = session_->SessionWrite(write_request, data);
  EXPECT_OK(write_result);
  EXPECT_OK(write_result.value().write().status());
  EXPECT_THAT(write_result.value().write().committed_size_bytes(),
              Eq(data.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");

  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_OK(finalize_response);
  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);

  ASSERT_OK(UnbundleRangeTracker(result_data));
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), result_data);
  ASSERT_OK(deserialized_agg);

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

  auto write_result1 = session_->SessionWrite(write_request1, blob1);
  EXPECT_OK(write_result1);
  EXPECT_OK(write_result1.value().write().status());
  EXPECT_THAT(write_result1.value().write().committed_size_bytes(),
              Eq(blob1.size()));
  auto write_result2 = session_->SessionWrite(write_request2, blob2);
  EXPECT_OK(write_result2);
  EXPECT_OK(write_result2.value().write().status());
  EXPECT_THAT(write_result2.value().write().committed_size_bytes(),
              Eq(blob2.size()));

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
  ASSERT_OK(deserialized_agg);

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_OK((*deserialized_agg)->Report(*checkpoint_builder));
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

  auto write_result1 = session_->SessionWrite(write_request1, blob1);
  EXPECT_OK(write_result1);
  EXPECT_OK(write_result1.value().write().status());
  EXPECT_THAT(write_result1.value().write().committed_size_bytes(),
              Eq(blob1.size()));
  auto write_result2 = session_->SessionWrite(write_request2, blob2);
  EXPECT_OK(write_result2);
  EXPECT_OK(write_result2.value().write().status());
  EXPECT_THAT(write_result2.value().write().committed_size_bytes(),
              Eq(blob2.size()));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  oak::crypto::v1::Signature fake_signature;
  fake_signature.set_signature("signature");
  EXPECT_CALL(*mock_signing_key_handle_, Sign(_))
      .WillOnce(Return(fake_signature));

  auto finalize_response = session_->FinalizeSession(finalize_request, unused);

  EXPECT_OK(finalize_response);
  ASSERT_TRUE(finalize_response->read().finish_read());
  auto actual_metadata = finalize_response->read().first_response_metadata();
  ASSERT_GT(actual_metadata.total_size_bytes(), 0);
  ASSERT_EQ(actual_metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  ASSERT_TRUE(actual_metadata.has_hpke_plus_aead_data());

  // Verify the release token.
  FinalResultConfiguration final_result_config;
  ASSERT_TRUE(finalize_response->read().first_response_configuration().UnpackTo(
      &final_result_config));
  absl::StatusOr<ReleaseToken> release_token =
      ReleaseToken::Decode(final_result_config.release_token());
  ASSERT_OK(release_token);
  EXPECT_EQ(release_token->encryption_key_id, "report");
  EXPECT_EQ(release_token->src_state, initial_private_state_);
  EXPECT_EQ(release_token->signature, "signature");
  BudgetState new_state;
  EXPECT_TRUE(new_state.ParseFromString(release_token->dst_state.value()));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "key_foo" budget: 0 }
                buckets { key: "key_bar" budget: 3 }
                buckets { key: "key_baz" budget: 0 }
              )pb"));

  std::string ciphertext = finalize_response->read().data();
  auto result_data = Decrypt(actual_metadata, ciphertext);

  auto parser = parser_factory.Create(absl::Cord(std::move(result_data)));
  EXPECT_OK(parser);
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
  EXPECT_OK(session_->SessionWrite(write_request, data));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
  EXPECT_OK(session_->SessionCommit(commit_request));

  // Do another write and commit with the same range again - it should fail
  // because it uses the same range.
  EXPECT_OK(session_->SessionWrite(write_request, data));
  EXPECT_THAT(session_->SessionCommit(commit_request),
              IsCode(absl::StatusCode::kFailedPrecondition));
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
  EXPECT_OK(session_->SessionWrite(write_request, data));

  CommitRequest commit_request;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 2 end: 3 }
  )pb");
  commit_request.mutable_configuration()->PackFrom(commit_config);
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
  EXPECT_OK(session_->SessionWrite(write_request1, blob1));
  EXPECT_THAT(session_->SessionWrite(write_request2, blob2),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KmsFedSqlSessionWriteTest, MergeReportBudgetExhausted) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_TRUE(input_aggregator->Accumulate(*input_parser).ok());

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

  EXPECT_OK(session_->SessionWrite(write_request1, blob1));
  EXPECT_OK(session_->SessionWrite(write_request2, blob2));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  FinalizeRequest finalize_request;
  finalize_request.mutable_configuration()->PackFrom(finalize_config);
  BlobMetadata unused;
  EXPECT_THAT(session_->FinalizeSession(finalize_request, unused),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
