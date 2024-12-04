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
#include "containers/tff_server/confidential_transform_server.h"

#include <execution>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/file_info.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::tff_server {

namespace {

using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::FileInfo;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::fcp::confidentialcompute::TffSessionWriteConfig;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::tensorflow_federated::v0::Value;
using ::testing::HasSubstr;

constexpr absl::string_view kNoArgumentComputationPath =
    "containers/tff_server/testing/no_argument_function.txtpb";
constexpr absl::string_view kServerDataComputationPath =
    "containers/tff_server/testing/server_data_function.txtpb";
constexpr absl::string_view kClientDataComputationPath =
    "containers/tff_server/testing/client_data_function.txtpb";

absl::StatusOr<Value> LoadFileAsTffValue(absl::string_view path) {
  // Before creating the std::ifstream, convert the absl::string_view to
  // std::string.
  std::string path_str(path);
  std::ifstream file_istream(path_str);
  if (!file_istream) {
    return absl::FailedPreconditionError("Error loading file: " + path_str);
  }
  std::stringstream file_stream;
  file_stream << file_istream.rdbuf();
  federated_language::Computation computation;
  if (!google::protobuf::TextFormat::ParseFromString(
          std::move(file_stream.str()), &computation)) {
    return absl::InvalidArgumentError(
        "Error parsing TFF Computation from file.");
  }
  Value value;
  *value.mutable_computation() = std::move(computation);
  return value;
}

std::string BuildClientCheckpoint(std::initializer_list<int32_t> input_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_INT32,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     CreateTestData<int32_t>(input_values));
  CHECK_OK(t1);
  CHECK_OK(ckpt_builder->Add("key", *t1));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

WriteRequest CreateDefaultWriteRequest(std::string data, bool client_upload) {
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  TffSessionWriteConfig config;
  config.set_uri("uri");
  config.set_client_upload(client_upload);
  WriteRequest write_request;
  *write_request.mutable_first_request_metadata() = metadata;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.set_commit(true);
  write_request.set_data(data);
  return write_request;
}

WriteRequest CreateDefaultClientWriteRequest(std::string data) {
  return CreateDefaultWriteRequest(data, true);
}

WriteRequest CreateWriteRequest(std::string data, FileInfo file_info) {
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  TffSessionWriteConfig config;
  config.set_uri(file_info.uri());
  config.set_client_upload(file_info.client_upload());
  WriteRequest write_request;
  *write_request.mutable_first_request_metadata() = metadata;
  write_request.mutable_first_request_configuration()->PackFrom(config);
  write_request.set_commit(true);
  write_request.set_data(data);
  return write_request;
}

TffSessionConfig DefaultSessionConfiguration() {
  TffSessionConfig config = PARSE_TEXT_PROTO(R"pb(
    num_clients: 3
    output_access_policy_node_id: 3
  )pb");
  *config.mutable_function() = *LoadFileAsTffValue(kNoArgumentComputationPath);
  return config;
}

TffSessionConfig CreateSessionConfiguration(Value function, Value argument,
                                            int num_clients) {
  TffSessionConfig config = PARSE_TEXT_PROTO(R"pb(
    output_access_policy_node_id: 3
  )pb");
  *config.mutable_function() = std::move(function);
  *config.mutable_initial_arg() = std::move(argument);
  config.set_num_clients(num_clients);
  return config;
}

TEST(TffConfidentialTransform, InitializeTransformSuccess) {
  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  TffConfidentialTransform service(&mock_crypto_stub);
  std::unique_ptr<Server> server;
  std::unique_ptr<ConfidentialTransform::Stub> stub;

  int port;
  const std::string server_address = "[::1]:";
  ServerBuilder builder;
  builder.AddListeningPort(server_address + "0",
                           grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service);
  server = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
            << std::endl;
  stub = ConfidentialTransform::NewStub(
      grpc::CreateChannel(server_address + std::to_string(port),
                          grpc::InsecureChannelCredentials()));

  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  auto status = stub->Initialize(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::OK);
}

TEST(TffSessionTest, ConfigureSessionSuccess) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());
}

TEST(TffSessionTest, SessionAlreadyConfiguredFailure) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  auto status = session.ConfigureSession(request);
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(), HasSubstr("Session already configured."));
}

TEST(TffSessionTest, InvalidConfigureRequestFailure) {
  TffSession session;
  SessionRequest request;
  auto status = session.ConfigureSession(request);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("TffSessionConfig invalid."));
}

TEST(TffSessionTest, WriteBeforeConfigureFailure) {
  TffSession session;
  auto status =
      session.SessionWrite(CreateDefaultClientWriteRequest("data"), "data")
          .status();
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(), HasSubstr("Session must be configured before"));
}

TEST(TffSessionTest, InvalidWriteConfigSessionWriteSuccessButDataIgnored) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());
  WriteRequest write_request = CreateDefaultClientWriteRequest("data");
  write_request.clear_first_request_configuration();
  auto response = session.SessionWrite(write_request, "data").value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(response.write().status().message(),
              HasSubstr("Failed to parse TffSessionWriteConfig."));
}

TEST(TffSessionTest, WriteInvalidDataSessionWriteSuccessButDataIgnored) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());
  WriteRequest write_request = CreateDefaultWriteRequest("data", false);
  auto response = session.SessionWrite(write_request, "data").value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(response.write().status().message(),
              HasSubstr("Failed to deserialize the data"));
}

TEST(TffSessionTest,
     WriteInvalidClientCheckpointWriteSessionSuccessButDataIgnored) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());
  WriteRequest write_request = CreateDefaultClientWriteRequest("data");
  auto response = session.SessionWrite(write_request, "data").value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(
      response.write().status().message(),
      HasSubstr("Failed to deserialize the federated compute checkpoint."));
}

TEST(TffSessionTest, WriteDataSameUriWriteSessionSuccessButDataIgnored) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  Value data;
  data.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(20);
  std::string data_string = data.SerializeAsString();
  WriteRequest write_request = CreateDefaultWriteRequest(data_string, false);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      response.write().status().message(),
      HasSubstr("Data corresponding to URI already written to session."));
}

TEST(TffSessionTest, WriteClientDataSameUriWriteSessionSuccessButDataIgnored) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  std::string data_string = BuildClientCheckpoint({1});
  WriteRequest write_request = CreateDefaultClientWriteRequest(data_string);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      response.write().status().message(),
      HasSubstr("Data corresponding to URI already written to session."));
}

TEST(TffSessionTest, WriteDataSuccess) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  Value data;
  data.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(20);
  std::string data_string = data.SerializeAsString();
  WriteRequest write_request = CreateDefaultWriteRequest(data_string, false);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);
}

TEST(TffSessionTest, WriteClientDataSuccess) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  std::string data_string = BuildClientCheckpoint({1});
  WriteRequest write_request = CreateDefaultClientWriteRequest(data_string);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);
}

TEST(TffSessionTest, FinalizeBeforeConfigureFailure) {
  TffSession session;
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest request;
  auto status = session.FinalizeSession(request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(), HasSubstr("Session must be configured"));
}

TEST(TffSessionTest, FinalizeInvalidFunctionFailure) {
  TffSession session;
  SessionRequest request;
  TffSessionConfig config = DefaultSessionConfiguration();
  config.clear_function();
  request.mutable_configure()->mutable_configuration()->PackFrom(config);
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto status = session.FinalizeSession(finalize_request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

TEST(TffSessionTest, FinalizeWithoutArgumentSuccess) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();
  ReadResponse read_response = result.read();
  EXPECT_EQ(read_response.first_response_metadata().compression_type(),
            BlobMetadata::COMPRESSION_TYPE_NONE);
  Value value;
  value.ParseFromString(read_response.data());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(value.federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  EXPECT_EQ(flat(0), 10) << flat(0);
}

TEST(TffSessionTest, FinalizeMultipleDataInputsSuccess) {
  FileInfo file_info_1;
  file_info_1.set_uri("server1");
  FileInfo file_info_2;
  file_info_2.set_uri("server2");
  FileInfo file_info_3;
  file_info_3.set_uri("server3");
  file_info_3.set_key("key3");

  // Create Function
  Value function = *LoadFileAsTffValue(kServerDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("server");
  argument.mutable_federated()->mutable_type()->set_all_equal(true);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_3);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 3));
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Write Data to Session
  Value data1;
  data1.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(10);
  std::string data_string = data1.SerializeAsString();
  WriteRequest write_request = CreateWriteRequest(data_string, file_info_1);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  Value data2;
  data2.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(20);
  data_string = data2.SerializeAsString();
  write_request = CreateWriteRequest(data_string, file_info_2);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  Value data3;
  data3.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(30);
  Value data4;
  data4.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(-1);
  Value data5;
  auto data_struct = data5.mutable_struct_();
  auto struct_v1 = data_struct->add_element();
  struct_v1->set_name("key3");
  *struct_v1->mutable_value() = data3;
  auto struct_v2 = data_struct->add_element();
  struct_v2->set_name("foo");
  *struct_v2->mutable_value() = data4;
  data_string = data5.SerializeAsString();
  write_request = CreateWriteRequest(data_string, file_info_3);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();
  ReadResponse read_response = result.read();
  Value value;
  value.ParseFromString(read_response.data());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(
          value.struct_().element(0).value().federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  // 30 scaled by 10 = 300
  EXPECT_EQ(flat(0), 300) << flat(0);
  output_tensor = tensorflow_federated::DeserializeTensorValue(
                      value.struct_().element(1).value().federated().value(0))
                      .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat2 = output_tensor.unaligned_flat<int32_t>();
  // 30 scaled by 10 * num clients = 900
  EXPECT_EQ(flat2(0), 900) << flat2(0);
}

TEST(TffSessionTest, FinalizeMultipleClientDataInputsSuccess) {
  FileInfo file_info_1;
  file_info_1.set_uri("client1");
  file_info_1.set_key("key");
  file_info_1.set_client_upload(true);
  FileInfo file_info_2;
  file_info_2.set_uri("client2");
  file_info_2.set_key("key");
  file_info_2.set_client_upload(true);
  FileInfo file_info_3;
  file_info_3.set_uri("client3");
  file_info_3.set_key("key");
  file_info_3.set_client_upload(true);

  // Create Function
  Value function = *LoadFileAsTffValue(kClientDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("clients");
  argument.mutable_federated()->mutable_type()->set_all_equal(false);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_1);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_2);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_3);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 3));
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Write Data to Session
  std::string data_string = BuildClientCheckpoint({10});
  WriteRequest write_request = CreateWriteRequest(data_string, file_info_1);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = BuildClientCheckpoint({20});
  write_request = CreateWriteRequest(data_string, file_info_2);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = BuildClientCheckpoint({30});
  write_request = CreateWriteRequest(data_string, file_info_3);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();
  ReadResponse read_response = result.read();
  Value value;
  value.ParseFromString(read_response.data());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(value.federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  EXPECT_EQ(flat(0), 60) << flat(0);
}

TEST(TffSessionTest, FinalizeIgnoresInvalidDataInputsSuccess) {
  FileInfo file_info_1;
  file_info_1.set_uri("server1");
  FileInfo file_info_2;
  file_info_2.set_uri("server2");
  FileInfo file_info_3;
  file_info_3.set_uri("server3");
  file_info_3.set_key("key3");

  // Create Function
  Value function = *LoadFileAsTffValue(kServerDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("server");
  argument.mutable_federated()->mutable_type()->set_all_equal(true);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_3);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 3));
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Write Data to Session
  Value data1;
  data1.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(10);
  std::string data_string = data1.SerializeAsString();
  WriteRequest write_request = CreateWriteRequest(data_string, file_info_1);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = "invalid data";
  write_request = CreateWriteRequest(data_string, file_info_2);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);

  Value data3;
  data3.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(30);
  Value data4;
  data4.mutable_computation()
      ->mutable_literal()
      ->mutable_value()
      ->mutable_int32_list()
      ->add_value(-1);
  Value data5;
  auto data_struct = data5.mutable_struct_();
  auto struct_v1 = data_struct->add_element();
  struct_v1->set_name("key3");
  *struct_v1->mutable_value() = data3;
  auto struct_v2 = data_struct->add_element();
  struct_v2->set_name("foo");
  *struct_v2->mutable_value() = data4;
  data_string = data5.SerializeAsString();
  write_request = CreateWriteRequest(data_string, file_info_3);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();
  ReadResponse read_response = result.read();
  Value value;
  value.ParseFromString(read_response.data());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(
          value.struct_().element(0).value().federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  // 30 scaled by 10 = 300
  EXPECT_EQ(flat(0), 300) << flat(0);
  output_tensor = tensorflow_federated::DeserializeTensorValue(
                      value.struct_().element(1).value().federated().value(0))
                      .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat2 = output_tensor.unaligned_flat<int32_t>();
  // 30 scaled by 10 * num clients = 900
  EXPECT_EQ(flat2(0), 900) << flat2(0);
}

TEST(TffSessionTest, FinalizeIgnoresInvalidClientDataInputsSuccess) {
  FileInfo file_info_1;
  file_info_1.set_uri("client1");
  file_info_1.set_key("key");
  file_info_1.set_client_upload(true);
  FileInfo file_info_2;
  file_info_2.set_uri("client2");
  file_info_2.set_key("key");
  file_info_2.set_client_upload(true);
  FileInfo file_info_3;
  file_info_3.set_uri("client3");
  file_info_3.set_key("key");
  file_info_3.set_client_upload(true);

  // Create Function
  Value function = *LoadFileAsTffValue(kClientDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("clients");
  argument.mutable_federated()->mutable_type()->set_all_equal(false);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_1);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_2);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 2));
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Write Data to Session
  std::string data_string = BuildClientCheckpoint({10});
  WriteRequest write_request = CreateWriteRequest(data_string, file_info_1);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = BuildClientCheckpoint({20});
  write_request = CreateWriteRequest(data_string, file_info_2);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = "Invalid data.";
  write_request = CreateWriteRequest(data_string, file_info_3);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();
  ReadResponse read_response = result.read();
  Value value;
  value.ParseFromString(read_response.data());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(value.federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  EXPECT_EQ(flat(0), 30) << flat(0);
}

TEST(TffSessionTest, FinalizeEncryptsOutputSuccess) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();
  fcp::confidential_compute::MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
  )pb");
  metadata.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      ciphertext_associated_data);
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(reencryption_public_key.value());
  FinalizeRequest finalize_request;
  auto result = session.FinalizeSession(finalize_request, metadata).value();

  ReadResponse read_response = result.read();
  ASSERT_TRUE(
      read_response.first_response_metadata().has_hpke_plus_aead_data());
  BlobMetadata::HpkePlusAeadMetadata result_metadata =
      read_response.first_response_metadata().hpke_plus_aead_data();
  absl::StatusOr<std::string> decrypted_result = decryptor.Decrypt(
      read_response.data(), result_metadata.ciphertext_associated_data(),
      result_metadata.encrypted_symmetric_key(),
      result_metadata.ciphertext_associated_data(),
      result_metadata.encapsulated_public_key());
  ASSERT_TRUE(decrypted_result.ok()) << decrypted_result.status();

  Value value;
  value.ParseFromString(decrypted_result.value());
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(value.federated().value(0))
          .value();
  EXPECT_EQ(output_tensor.NumElements(), 1);
  auto flat = output_tensor.unaligned_flat<int32_t>();
  EXPECT_EQ(flat(0), 10) << flat(0);
}

TEST(TffSessionTest, FinalizeInvalidClientTensorFailure) {
  FileInfo file_info_1;
  file_info_1.set_uri("client1");
  file_info_1.set_key("key");
  file_info_1.set_client_upload(true);

  // Create Function
  Value function = *LoadFileAsTffValue(kClientDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("clients");
  argument.mutable_federated()->mutable_type()->set_all_equal(false);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_1);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 1));

  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Create checkpoint tensor with invalid key.
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_INT32, TensorShape({static_cast<int32_t>(1)}),
                     CreateTestData<int32_t>({1}));
  CHECK_OK(t1);
  CHECK_OK(ckpt_builder->Add("invalid_key", *t1));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);

  WriteRequest write_request =
      CreateDefaultClientWriteRequest(checkpoint_string);
  auto response =
      session.SessionWrite(write_request, checkpoint_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), checkpoint_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto status = session.FinalizeSession(finalize_request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("Data in argument was not provided"));
}

TEST(TffSessionTest, FinalizeDataNotReplaceableFailure) {
  FileInfo file_info_1;
  file_info_1.set_uri("client1");
  file_info_1.set_key("key");
  file_info_1.set_client_upload(true);
  FileInfo file_info_2;
  file_info_2.set_uri("client2");
  file_info_2.set_key("key");
  file_info_2.set_client_upload(true);
  FileInfo file_info_3;
  file_info_3.set_uri("client3");
  file_info_3.set_key("key");
  file_info_3.set_client_upload(true);

  // Create Function
  Value function = *LoadFileAsTffValue(kClientDataComputationPath);

  // Create Argument
  Value argument;
  argument.mutable_federated()
      ->mutable_type()
      ->mutable_placement()
      ->mutable_value()
      ->set_uri("clients");
  argument.mutable_federated()->mutable_type()->set_all_equal(false);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_1);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_2);
  argument.mutable_federated()
      ->add_value()
      ->mutable_computation()
      ->mutable_data()
      ->mutable_content()
      ->PackFrom(file_info_3);

  // Configure Session
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      CreateSessionConfiguration(std::move(function), std::move(argument), 3));
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  // Write Data to Session
  std::string data_string = BuildClientCheckpoint({10});
  WriteRequest write_request = CreateWriteRequest(data_string, file_info_1);
  auto response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = BuildClientCheckpoint({20});
  write_request = CreateWriteRequest(data_string, file_info_2);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), data_string.size());
  ASSERT_EQ(response.write().status().code(), grpc::StatusCode::OK);

  data_string = "Invalid data.";
  write_request = CreateWriteRequest(data_string, file_info_3);
  response = session.SessionWrite(write_request, data_string).value();
  ASSERT_EQ(response.write().committed_size_bytes(), 0);
  ASSERT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);

  // Finalize Session
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  FinalizeRequest finalize_request;
  auto status = session.FinalizeSession(finalize_request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("Data in argument was not provided"));
}

TEST(TffSessionTest, FinalizeEncryptWithInvalidBlobHeaderFailure) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  fcp::confidential_compute::MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
  )pb");
  metadata.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      "invalid ciphertext_associated_data");
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(reencryption_public_key.value());
  FinalizeRequest finalize_request;
  auto status = session.FinalizeSession(finalize_request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("Failed to parse the BlobHeader"));
}

TEST(TffSessionTest, FinalizeEncryptOutputRecordErrorFailure) {
  TffSession session;
  SessionRequest request;
  request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSessionConfiguration());
  ASSERT_TRUE(session.ConfigureSession(request).ok());

  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
  )pb");
  metadata.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      ciphertext_associated_data);
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key("invalid key");
  FinalizeRequest finalize_request;
  auto status = session.FinalizeSession(finalize_request, metadata).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("failed to decode CWT"));
}

}  // namespace

}  // namespace confidential_federated_compute::tff_server
