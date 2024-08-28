/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "containers/tff_worker/pipeline_transform_server.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_worker_configuration.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::tff_worker {

namespace {

namespace tff_proto = ::tensorflow_federated::v0;

using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::TffWorkerConfiguration;
using ::fcp::confidentialcompute::TffWorkerConfiguration_Aggregation;
using ::fcp::confidentialcompute::TffWorkerConfiguration_ClientWork;
using ::fcp::confidentialcompute::
    TffWorkerConfiguration_ClientWork_FedSqlTensorflowCheckpoint;
using ::fcp::confidentialcompute::
    TffWorkerConfiguration_ClientWork_FedSqlTensorflowCheckpoint_FedSqlColumn;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ClientContext;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::StatusCode;
using ::tensorflow_federated::kClientsUri;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::HasSubstr;
using ::testing::Test;

constexpr absl::string_view kAggregationComputationPath =
    "containers/tff_worker/testing/aggregation_computation.txtpb";
constexpr absl::string_view kClientWorkComputationPath =
    "containers/tff_worker/testing/client_work_computation.txtpb";

absl::StatusOr<tff_proto::Computation> LoadFileAsTffComputation(
    absl::string_view path) {
  // Before creating the std::ifstream, convert the absl::string_view to
  // std::string.
  std::string path_str(path);
  std::ifstream file_istream(path_str);
  if (!file_istream) {
    return absl::FailedPreconditionError("Error loading file: " + path_str);
  }
  std::stringstream file_stream;
  file_stream << file_istream.rdbuf();
  tff_proto::Computation computation;
  if (!google::protobuf::TextFormat::ParseFromString(
          std::move(file_stream.str()), &computation)) {
    return absl::InvalidArgumentError(
        "Error parsing TFF Computation from file.");
  }
  return std::move(computation);
}

tff_proto::Value BuildFederatedIntClientValue(float int_value) {
  tff_proto::Value value;
  tff_proto::Value_Federated* federated = value.mutable_federated();
  tensorflow_federated::v0::FederatedType* type_proto =
      federated->mutable_type();
  type_proto->set_all_equal(true);
  *type_proto->mutable_placement()->mutable_value()->mutable_uri() =
      kClientsUri;
  tensorflow::TensorShape shape({static_cast<int64_t>(1)});
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, shape);
  auto flat = tensor.flat<float>();
  flat(0) = int_value;
  tensorflow_federated::v0::Value* federated_value = federated->add_value();
  tensorflow_federated::SerializeTensorValue(tensor, federated_value);
  return value;
}

std::string BuildSingleInt64TensorCheckpoint(
    std::string column_name, std::initializer_list<float> input_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_FLOAT,
                     TensorShape({static_cast<int64_t>(input_values.size())}),
                     CreateTestData<float>(input_values));
  CHECK_OK(t1);
  CHECK_OK(ckpt_builder->Add(column_name, *t1));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

class TffPipelineTransformTest : public Test {
 public:
  void SetUp() override {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on "
              << server_address + std::to_string(port);
    channel_ =
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::experimental::LocalCredentials(LOCAL_TCP));
    stub_ = PipelineTransform::NewStub(channel_);
  }

  void TearDown() override { server_->Shutdown(); }

 protected:
  TffPipelineTransform service_;
  std::unique_ptr<Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
};

TEST_F(TffPipelineTransformTest, InvalidConfigureAndAttestRequest) {
  ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), HasSubstr("must contain configuration"));
}

TEST_F(TffPipelineTransformTest,
       MissingTffWorkerConfigurationConfigureAndAttestRequest) {
  ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  TffWorkerConfiguration_ClientWork wrong_proto;
  request.mutable_configuration()->PackFrom(wrong_proto);
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      HasSubstr(
          "must be a tff_worker_configuration_pb2.TffWorkerConfiguration"));
}

TEST_F(TffPipelineTransformTest, ConfigureAndAttestMoreThanOnce) {
  ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  // Populate test TffWorkerConfiguration with a computation that has client
  // work.
  TffWorkerConfiguration tff_worker_configuration;
  TffWorkerConfiguration_ClientWork* client_work =
      tff_worker_configuration.mutable_client_work();
  absl::StatusOr<tff_proto::Computation> computation_proto =
      LoadFileAsTffComputation(kClientWorkComputationPath);
  ASSERT_TRUE(computation_proto.ok()) << computation_proto.status();
  std::string serialized_computation;
  (*computation_proto).SerializeToString(&serialized_computation);
  client_work->set_serialized_client_work_computation(serialized_computation);
  request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto first_status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(first_status.ok()) << first_status.error_message();

  ClientContext second_context;
  auto second_status =
      stub_->ConfigureAndAttest(&second_context, request, &response);
  EXPECT_EQ(second_status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_THAT(second_status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(TffPipelineTransformTest, ValidConfigureAndAttest) {
  ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  // Populate test TffWorkerConfiguration with a computation that has client
  // work.
  TffWorkerConfiguration tff_worker_configuration;
  TffWorkerConfiguration_ClientWork* client_work =
      tff_worker_configuration.mutable_client_work();
  absl::StatusOr<tff_proto::Computation> computation_proto =
      LoadFileAsTffComputation(kClientWorkComputationPath);
  ASSERT_TRUE(computation_proto.ok()) << computation_proto.status();
  std::string serialized_computation;
  (*computation_proto).SerializeToString(&serialized_computation);
  client_work->set_serialized_client_work_computation(serialized_computation);
  request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_TRUE(status.ok()) << status.error_message();
}

TEST_F(TffPipelineTransformTest, TransformBeforeConfigureAndAttest) {
  ClientContext context;
  TransformRequest request;
  TransformResponse response;
  auto status = stub_->Transform(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest must be called before Transform"));
}

TEST_F(TffPipelineTransformTest, TransformCannotExecuteAggregation) {
  ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;

  TffWorkerConfiguration tff_worker_configuration;

  // Populate test TffWorkerConfiguration with a computation that has
  // aggregation work.
  TffWorkerConfiguration_Aggregation* aggregation =
      tff_worker_configuration.mutable_aggregation();
  absl::StatusOr<tff_proto::Computation> computation_proto =
      LoadFileAsTffComputation(kAggregationComputationPath);
  ASSERT_TRUE(computation_proto.ok()) << computation_proto.status();
  std::string serialized_computation;
  (*computation_proto).SerializeToString(&serialized_computation);
  aggregation->set_serialized_client_to_server_aggregation_computation(
      serialized_computation);
  configure_request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto configure_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_status.ok()) << configure_status.error_message();

  ClientContext transform_context;
  TransformRequest transform_request;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);
  ASSERT_EQ(transform_status.error_code(), grpc::UNIMPLEMENTED);
  ASSERT_THAT(transform_status.error_message(),
              HasSubstr("has not yet implemented"));
}

TEST_F(TffPipelineTransformTest, TransformExecutesClientWork) {
  ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;

  TffWorkerConfiguration tff_worker_configuration;

  // Populate test TffWorkerConfiguration with a computation that has client
  // work.
  TffWorkerConfiguration_ClientWork* client_work =
      tff_worker_configuration.mutable_client_work();
  absl::StatusOr<tff_proto::Computation> computation_proto =
      LoadFileAsTffComputation(kClientWorkComputationPath);
  ASSERT_TRUE(computation_proto.ok()) << computation_proto.status();
  std::string serialized_computation;
  (*computation_proto).SerializeToString(&serialized_computation);
  client_work->set_serialized_client_work_computation(serialized_computation);

  // Populate test TffWorkerConfiguration with a serialized broadcasted TFF
  // Value.
  std::string serialized_value;
  BuildFederatedIntClientValue(10).SerializeToString(&serialized_value);
  client_work->set_serialized_broadcasted_data(serialized_value);
  TffWorkerConfiguration_ClientWork_FedSqlTensorflowCheckpoint_FedSqlColumn*
      column = client_work->mutable_fed_sql_tensorflow_checkpoint()
                   ->add_fed_sql_columns();
  column->set_name("whimsy");
  column->set_data_type(tensorflow::DT_INT32);
  configure_request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto configure_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_status.ok()) << configure_status.error_message();

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt64TensorCheckpoint("whimsy", {1, 2, 3}));
  ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);
  ASSERT_TRUE(transform_status.ok()) << transform_status.error_message();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  tff_proto::Value value;
  value.ParseFromString(transform_response.outputs(0).unencrypted_data());
  EXPECT_TRUE(value.has_federated());
  EXPECT_EQ(value.federated().type().placement().value().uri(), kClientsUri);
  EXPECT_EQ(value.federated().value_size(), 1);
  EXPECT_TRUE(value.federated().value(0).has_struct_());
  EXPECT_EQ(value.federated().value(0).struct_().element_size(), 1);
  EXPECT_TRUE(
      value.federated().value(0).struct_().element(0).value().has_array());
  absl::StatusOr<tensorflow::Tensor> output_tensor =
      tensorflow_federated::DeserializeTensorValue(
          value.federated().value(0).struct_().element(0).value());
  EXPECT_TRUE(output_tensor.ok());
  EXPECT_EQ(output_tensor.value().NumElements(), 3);

  // Test client work computation adds the broadcasted value to each of the
  // values in the input tensor.
  auto flat = output_tensor.value().unaligned_flat<float>();
  EXPECT_EQ(flat(0), 11);
  EXPECT_EQ(flat(1), 12);
  EXPECT_EQ(flat(2), 13);
}

}  // namespace

}  // namespace confidential_federated_compute::tff_worker
