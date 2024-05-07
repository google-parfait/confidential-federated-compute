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
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "containers/agg_core/pipeline_transform_server.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
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
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::agg_core {

namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::GenerateNoncesRequest;
using ::fcp::confidentialcompute::GenerateNoncesResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::google::protobuf::RepeatedPtrField;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
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

class AggCoreTransformTest : public benchmark::Fixture {
 public:
  void SetUp(::benchmark::State& state) override {
    int port;
    service_ = std::make_unique<AggCorePipelineTransform>(&mock_crypto_stub_);
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    channel_ = grpc::CreateChannel(server_address + std::to_string(port),
                                   grpc::InsecureChannelCredentials());
    stub_ = PipelineTransform::NewStub(channel_);
  }

  void TearDown(::benchmark::State& state) override {
    stub_.reset(nullptr);
    channel_.reset();
    server_->Shutdown();
    server_.reset(nullptr);
    service_.reset(nullptr);
  }

 protected:
  // Returns default configuration.
  Configuration DefaultConfiguration() const;

  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  std::unique_ptr<AggCorePipelineTransform> service_ = nullptr;
  std::unique_ptr<Server> server_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
  std::shared_ptr<grpc::Channel> channel_ = nullptr;
};

Configuration AggCoreTransformTest::DefaultConfiguration() const {
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

BENCHMARK_DEFINE_F(AggCoreTransformTest, BM_Transform10000Records)
(benchmark::State& state) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  grpc::Status config_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  if (!config_status.ok()) {
    LOG(INFO) << config_status.error_message();
  }
  ASSERT_TRUE(config_status.ok());

  TransformRequest transform_request;
  for (int i = 0; i < 10000; i++) {
    transform_request.add_inputs()->set_unencrypted_data(
        BuildSingleInt32TensorCheckpoint("foo", {i}));
  }
  for (auto& input : *transform_request.mutable_inputs()) {
    input.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  }
  for (auto _ : state) {
    grpc::ClientContext transform_context;
    TransformResponse transform_response;

    ASSERT_TRUE(stub_
                    ->Transform(&transform_context, transform_request,
                                &transform_response)
                    .ok());
  }
}
BENCHMARK_REGISTER_F(AggCoreTransformTest, BM_Transform10000Records);

}  // namespace

}  // namespace confidential_federated_compute::agg_core

BENCHMARK_MAIN();
