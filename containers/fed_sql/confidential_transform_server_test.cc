// Copyright 2024 Google LLC.
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
#include "containers/fed_sql/confidential_transform_server.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/agg_core_container_config.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::AggCoreAggregationType;
using ::fcp::confidentialcompute::AggCoreContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::AggCoreContainerWriteConfiguration;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::RepeatedPtrField;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::Test;
using testing::UnorderedElementsAre;

inline constexpr int kMaxNumSessions = 8;
inline constexpr long kMaxSessionMemoryBytes = 1000000;

std::string BuildSingleInt32TensorCheckpoint(
    std::string column_name, std::initializer_list<int32_t> input_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_INT32,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     CreateTestData<int32_t>(input_values));
  CHECK_OK(t1);
  CHECK_OK(ckpt_builder->Add(column_name, *t1));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

SessionRequest CreateDefaultWriteRequest(AggCoreAggregationType agg_type,
                                         std::string data) {
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  AggCoreContainerWriteConfiguration config;
  config.set_type(agg_type);
  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->mutable_first_request_configuration()->PackFrom(config);
  write_request->set_commit(true);
  write_request->set_data(data);
  return request;
}

class FedSqlServerTest : public Test {
 public:
  FedSqlServerTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~FedSqlServerTest() override { server_->Shutdown(); }

 protected:
  // Returns default configuration.
  Configuration DefaultConfiguration() const;

  // Returns the default BlobMetadata
  BlobMetadata DefaultBlobMetadata() const;

  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  FedSqlConfidentialTransform service_{&mock_crypto_stub_, kMaxNumSessions,
                                       kMaxSessionMemoryBytes};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
};

Configuration FedSqlServerTest::DefaultConfiguration() const {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 1 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 1 }
      }
    }
  )pb");
}

TEST_F(FedSqlServerTest, InvalidInitializeRequest) {
  grpc::ClientContext context;
  Configuration invalid_config;
  invalid_config.add_intrinsic_configs()->set_intrinsic_uri("BAD URI");
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  auto status = stub_->Initialize(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("is not a supported intrinsic_uri"));
}

TEST_F(FedSqlServerTest, InitializeRequestWrongMessageType) {
  grpc::ClientContext context;
  google::protobuf::Value value;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(value);

  auto status = stub_->Initialize(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration cannot be unpacked."));
}

TEST_F(FedSqlServerTest, InitializeRequestNoIntrinsicConfigs) {
  grpc::ClientContext context;
  Configuration invalid_config;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  auto status = stub_->Initialize(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration must have exactly one IntrinsicConfig"));
}

TEST_F(FedSqlServerTest, FedSqlDpGroupByInvalidParametersInitialize) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  Configuration fedsql_dp_group_by_config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args { parameter { dtype: DT_INT64 int64_val: 42 } }
      intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 2.2 } }
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
  request.mutable_configuration()->PackFrom(fedsql_dp_group_by_config);

  auto status = stub_->Initialize(&context, request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr(
          "`fedsql_dp_group_by` parameters must both have type DT_DOUBLE"));
}

TEST_F(FedSqlServerTest, MultipleTopLevelIntrinsicsInitialize) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  Configuration fedsql_dp_group_by_config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "federated_sum"
      output_tensors {
        name: "key_out"
        dtype: DT_INT64
        shape { dim_sizes: -1 }
      }
    }
    intrinsic_configs: {
      intrinsic_uri: "federated_sum"
      output_tensors {
        name: "key_out"
        dtype: DT_INT64
        shape { dim_sizes: -1 }
      }
    }
  )pb");
  request.mutable_configuration()->PackFrom(fedsql_dp_group_by_config);

  auto status = stub_->Initialize(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration must have exactly one IntrinsicConfig"));
}

TEST_F(FedSqlServerTest, InitializeMoreThanOnce) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->Initialize(&context, request, &response).ok());

  grpc::ClientContext second_context;
  auto status = stub_->Initialize(&second_context, request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Initialize can only be called once"));
}

TEST_F(FedSqlServerTest, ValidInitialize) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->Initialize(&context, request, &response).ok());
}

TEST_F(FedSqlServerTest, FedSqlDpGroupByInitializeGeneratesConfigProperties) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  Configuration fedsql_dp_group_by_config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 1.1 } }
      intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 2.2 } }
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
  request.mutable_configuration()->PackFrom(fedsql_dp_group_by_config);

  auto status = stub_->Initialize(&context, request, &response);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_TRUE(cwt.ok());
  ASSERT_EQ(cwt->config_properties.fields().at("intrinsic_uri").string_value(),
            "fedsql_dp_group_by");
  ASSERT_EQ(cwt->config_properties.fields().at("epsilon").number_value(), 1.1);
  ASSERT_EQ(cwt->config_properties.fields().at("delta").number_value(), 2.2);
}

TEST_F(FedSqlServerTest, SessionConfigureGeneratesNonce) {
  grpc::ClientContext configure_context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->Initialize(&configure_context, request, &response).ok());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);
  ASSERT_EQ(session_response.configure().write_capacity_bytes(),
            kMaxSessionMemoryBytes);
}

TEST_F(FedSqlServerTest, SessionRejectsMoreThanMaximumNumSessions) {
  grpc::ClientContext configure_context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->Initialize(&configure_context, request, &response).ok());

  std::vector<std::unique_ptr<
      ::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>>
      streams;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
  for (int i = 0; i < kMaxNumSessions; i++) {
    std::unique_ptr<grpc::ClientContext> session_context =
        std::make_unique<grpc::ClientContext>();
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure();

    std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
        stream = stub_->Session(session_context.get());
    ASSERT_TRUE(stream->Write(session_request));
    ASSERT_TRUE(stream->Read(&session_response));

    // Keep the context and stream so they don't go out of scope and end the
    // session.
    contexts.emplace_back(std::move(session_context));
    streams.emplace_back(std::move(stream));
  }

  grpc::ClientContext rejected_context;
  SessionRequest rejected_request;
  SessionResponse rejected_response;
  rejected_request.mutable_configure();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&rejected_context);
  ASSERT_TRUE(stream->Write(rejected_request));
  ASSERT_FALSE(stream->Read(&rejected_response));
  ASSERT_EQ(stream->Finish().error_code(), grpc::StatusCode::UNAVAILABLE);
}

TEST_F(FedSqlServerTest, SessionBeforeInitialize) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  ASSERT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

std::string BuildFedSqlGroupByCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<uint64_t> val_col_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<uint64_t>(key_col_values));
  absl::StatusOr<Tensor> val =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(val_col_values.size())}),
                     CreateTestData<uint64_t>(val_col_values));
  CHECK_OK(key);
  CHECK_OK(val);
  CHECK_OK(ckpt_builder->Add("key", *key));
  CHECK_OK(ckpt_builder->Add("val", *val));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

TEST_F(FedSqlServerTest, TransformExecutesFedSqlGroupBy) {
  grpc::ClientContext init_context;
  InitializeRequest request;
  InitializeResponse response;
  Configuration fedsql_config = PARSE_TEXT_PROTO(R"pb(
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
  request.mutable_configuration()->PackFrom(fedsql_config);

  ASSERT_TRUE(stub_->Initialize(&init_context, request, &response).ok());

  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->mutable_configuration();
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));

  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({1, 1, 2}, {1, 2, 5}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  SessionRequest write_request_2 =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}));
  SessionResponse write_response_2;
  ASSERT_TRUE(stream->Write(write_request_2));
  ASSERT_TRUE(stream->Read(&write_response_2));

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query sums the val column, grouping by key
  ASSERT_EQ(col_values->num_elements(), 3);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_THAT(col_values->AsSpan<int64_t>(), UnorderedElementsAre(7, 5, 0));
}

class FedSqlServerFederatedSumTest : public FedSqlServerTest {
 public:
  FedSqlServerFederatedSumTest() {
    grpc::ClientContext configure_context;
    InitializeRequest request;
    InitializeResponse response;
    request.mutable_configuration()->PackFrom(DefaultConfiguration());

    CHECK(stub_->Initialize(&configure_context, request, &response).ok());
    public_key_ = response.public_key();

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure();

    stream_ = stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
    session_nonce_ = session_response.configure().nonce();
  }

 protected:
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::string session_nonce_;
  std::string public_key_;
};

TEST_F(FedSqlServerFederatedSumTest, SessionWriteAccumulateCommitsBlob) {
  FederatedComputeCheckpointParserFactory parser_factory;

  std::string data = BuildSingleInt32TensorCheckpoint("foo", {1});
  SessionRequest write_request =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE, data);
  SessionResponse write_response;

  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  ASSERT_TRUE(write_response.has_write());
  ASSERT_EQ(write_response.write().committed_size_bytes(), data.size());
  ASSERT_EQ(write_response.write().status().code(), grpc::OK);
  ASSERT_EQ(write_response.write().write_capacity_bytes(),
            kMaxSessionMemoryBytes);
}

TEST_F(FedSqlServerFederatedSumTest, SessionAccumulatesAndReports) {
  SessionRequest write_request =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildSingleInt32TensorCheckpoint("foo", {1}));
  SessionResponse write_response;

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 2);
}

TEST_F(FedSqlServerFederatedSumTest, SessionAccumulatesAndSerializes) {
  SessionRequest write_request =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildSingleInt32TensorCheckpoint("foo", {1}));
  SessionResponse write_response;

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(),
                                        finalize_response.read().data());
  ASSERT_TRUE(deserialized_agg.ok());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_TRUE((*deserialized_agg)->Report(*checkpoint_builder).ok());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 2);
}

TEST_F(FedSqlServerFederatedSumTest, SessionMergesAndReports) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory
          .Create(absl::Cord(BuildSingleInt32TensorCheckpoint("foo", {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_TRUE(input_aggregator->Accumulate(*input_parser).ok());

  SessionRequest write_request = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_MERGE,
      (std::move(*input_aggregator).Serialize()).value());
  SessionResponse write_response;

  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 3);
}

TEST_F(FedSqlServerFederatedSumTest, SessionIgnoresUnparseableInputs) {
  SessionRequest write_request_1 =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildSingleInt32TensorCheckpoint("foo", {7}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));

  SessionRequest write_request_2 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, "invalid checkpoint");
  SessionResponse write_response_2;

  ASSERT_TRUE(stream_->Write(write_request_2));
  ASSERT_TRUE(stream_->Read(&write_response_2));

  ASSERT_TRUE(write_response_2.has_write());
  ASSERT_EQ(write_response_2.write().committed_size_bytes(), 0);
  ASSERT_EQ(write_response_2.write().status().code(), grpc::INVALID_ARGUMENT);
  ASSERT_EQ(write_response_2.write().write_capacity_bytes(),
            kMaxSessionMemoryBytes);

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The invalid input should be ignored
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 7);
}

TEST_F(FedSqlServerFederatedSumTest, SessionFailsIfInputCannotBeAccumulated) {
  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildSingleInt32TensorCheckpoint("bad_col_name", {7}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_FALSE(stream_->Read(&write_response_1));
  ASSERT_EQ(stream_->Finish().error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(FedSqlServerFederatedSumTest, SessionDecryptsMultipleRecordsAndReports) {
  std::string input_col_name = "foo";
  std::string output_col_name = "foo_out";

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildSingleInt32TensorCheckpoint(input_col_name, {1});
  uint32_t counter_0 = 0;
  std::string nonce_0(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_0.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_0.data() + session_nonce_.length(), &counter_0,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_0 =
      crypto_test_utils::CreateRewrappedRecord(
          message_0, ciphertext_associated_data, public_key_, nonce_0,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_0.ok()) << rewrapped_record_0.status();

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  AggCoreContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_0);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(
      rewrapped_record_0->hpke_plus_aead_data().ciphertext());

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::OK);

  std::string message_1 = BuildSingleInt32TensorCheckpoint(input_col_name, {2});
  uint32_t counter_1 = 1;
  std::string nonce_1(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_1.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_1.data() + session_nonce_.length(), &counter_1,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, public_key_, nonce_1,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_1);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(
      rewrapped_record_1->hpke_plus_aead_data().ciphertext());

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  std::string message_2 = BuildSingleInt32TensorCheckpoint(input_col_name, {3});
  uint32_t counter_2 = 2;
  std::string nonce_2(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_2.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_2.data() + session_nonce_.length(), &counter_2,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_2 =
      crypto_test_utils::CreateRewrappedRecord(
          message_2, ciphertext_associated_data, public_key_, nonce_2,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_2.ok()) << rewrapped_record_2.status();

  SessionRequest request_2;
  WriteRequest* write_request_2 = request_2.mutable_write();
  *write_request_2->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_2);
  write_request_2->mutable_first_request_configuration()->PackFrom(config);
  write_request_2->set_commit(true);
  write_request_2->set_data(
      rewrapped_record_2->hpke_plus_aead_data().ciphertext());

  SessionResponse response_2;

  ASSERT_TRUE(stream_->Write(request_2));
  ASSERT_TRUE(stream_->Read(&response_2));
  ASSERT_EQ(response_2.write().status().code(), grpc::OK);

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 6);
}

TEST_F(FedSqlServerFederatedSumTest,
       SessionDecryptsMultipleRecordsAndSerializes) {
  std::string input_col_name = "foo";
  std::string output_col_name = "foo_out";

  MessageDecryptor earliest_decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      earliest_decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());

  absl::StatusOr<OkpCwt> reencryption_okp_cwt =
      OkpCwt::Decode(*reencryption_public_key);
  ASSERT_TRUE(reencryption_okp_cwt.ok());
  reencryption_okp_cwt->expiration_time = absl::FromUnixSeconds(1);
  absl::StatusOr<std::string> earliest_reencryption_key =
      reencryption_okp_cwt->Encode();
  ASSERT_TRUE(earliest_reencryption_key.ok());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildSingleInt32TensorCheckpoint(input_col_name, {1});
  uint32_t counter_0 = 0;
  std::string nonce_0(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_0.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_0.data() + session_nonce_.length(), &counter_0,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_0 =
      crypto_test_utils::CreateRewrappedRecord(
          message_0, ciphertext_associated_data, public_key_, nonce_0,
          *earliest_reencryption_key);
  ASSERT_TRUE(rewrapped_record_0.ok()) << rewrapped_record_0.status();

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  AggCoreContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_0);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(
      rewrapped_record_0->hpke_plus_aead_data().ciphertext());

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::OK);

  std::string message_1 = BuildSingleInt32TensorCheckpoint(input_col_name, {2});
  uint32_t counter_1 = 1;
  std::string nonce_1(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_1.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_1.data() + session_nonce_.length(), &counter_1,
              sizeof(uint32_t));

  MessageDecryptor later_decryptor;
  absl::StatusOr<std::string> other_reencryption_public_key =
      later_decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(other_reencryption_public_key.ok());

  absl::StatusOr<OkpCwt> other_reencryption_okp_cwt =
      OkpCwt::Decode(*other_reencryption_public_key);
  ASSERT_TRUE(other_reencryption_okp_cwt.ok());
  other_reencryption_okp_cwt->expiration_time = absl::FromUnixSeconds(99);
  absl::StatusOr<std::string> later_reencryption_key =
      reencryption_okp_cwt->Encode();
  ASSERT_TRUE(later_reencryption_key.ok());
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, public_key_, nonce_1,
          *later_reencryption_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_1);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(
      rewrapped_record_1->hpke_plus_aead_data().ciphertext());

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  // Unencrypted request should be incorporated, but the serialized result
  // should still be encrypted.
  SessionRequest unencrypted_request =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildSingleInt32TensorCheckpoint("foo", {3}));
  SessionResponse unencrypted_response;

  ASSERT_TRUE(stream_->Write(unencrypted_request));
  ASSERT_TRUE(stream_->Read(&unencrypted_response));
  ASSERT_EQ(unencrypted_response.write().status().code(), grpc::OK);

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  EXPECT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(finalize_response.read()
                  .first_response_metadata()
                  .has_hpke_plus_aead_data());

  BlobMetadata::HpkePlusAeadMetadata result_metadata =
      finalize_response.read().first_response_metadata().hpke_plus_aead_data();
  // The decryptor with the earliest set expiration time should be able to
  // decrypt the encrypted results. The later decryptor should not.
  absl::StatusOr<std::string> decrypted_result =
      earliest_decryptor.Decrypt(finalize_response.read().data(),
                                 result_metadata.ciphertext_associated_data(),
                                 result_metadata.encrypted_symmetric_key(),
                                 result_metadata.ciphertext_associated_data(),
                                 result_metadata.encapsulated_public_key());
  ASSERT_TRUE(decrypted_result.ok()) << decrypted_result.status();
  absl::StatusOr<std::string> failed_decrypt =
      later_decryptor.Decrypt(finalize_response.read().data(),
                              result_metadata.ciphertext_associated_data(),
                              result_metadata.encrypted_symmetric_key(),
                              result_metadata.ciphertext_associated_data(),
                              result_metadata.encapsulated_public_key());
  ASSERT_FALSE(failed_decrypt.ok()) << failed_decrypt.status();

  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(),
                                        *decrypted_result);
  ASSERT_TRUE(deserialized_agg.ok());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_TRUE((*deserialized_agg)->Report(*checkpoint_builder).ok());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 6);
}

TEST_F(FedSqlServerFederatedSumTest, TransformIgnoresUndecryptableInputs) {
  std::string input_col_name = "foo";
  std::string output_col_name = "foo_out";

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data = "ciphertext associated data";

  // Create one record that will fail to decrypt and one record that can be
  // successfully decrypted.
  std::string message_0 = BuildSingleInt32TensorCheckpoint(input_col_name, {1});
  uint32_t counter_0 = 0;
  std::string nonce_0(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_0.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_0.data() + session_nonce_.length(), &counter_0,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_0 =
      crypto_test_utils::CreateRewrappedRecord(
          message_0, ciphertext_associated_data, public_key_, nonce_0,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_0.ok()) << rewrapped_record_0.status();

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  AggCoreContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_0);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data("undecryptable");

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::INVALID_ARGUMENT);

  std::string message_1 = BuildSingleInt32TensorCheckpoint(input_col_name, {2});
  uint32_t counter_1 = 1;
  std::string nonce_1(session_nonce_.length() + sizeof(uint32_t), '\0');
  std::memcpy(nonce_1.data(), session_nonce_.data(), session_nonce_.length());
  std::memcpy(nonce_1.data() + session_nonce_.length(), &counter_1,
              sizeof(uint32_t));
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, public_key_, nonce_1,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_1);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(
      rewrapped_record_1->hpke_plus_aead_data().ciphertext());

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  AggCoreContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The undecryptable write is ignored, and only the valid write is aggregated.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 2);
}

}  // namespace

}  // namespace confidential_federated_compute::fed_sql
