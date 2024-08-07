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
#include "containers/agg_core/pipeline_transform_server.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
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

class AggCoreTransformTest : public Test {
 public:
  AggCoreTransformTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = PipelineTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~AggCoreTransformTest() override { server_->Shutdown(); }

 protected:
  // Returns default configuration.
  Configuration DefaultConfiguration() const;

  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  AggCorePipelineTransform service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
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

TEST_F(AggCoreTransformTest, InvalidConfigureAndAttestRequest) {
  grpc::ClientContext context;
  Configuration invalid_config;
  invalid_config.add_intrinsic_configs()->set_intrinsic_uri("BAD URI");
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("is not a supported intrinsic_uri"));
}

TEST_F(AggCoreTransformTest, ConfigureAndAttestRequestWrongMessageType) {
  grpc::ClientContext context;
  google::protobuf::Value value;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(value);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration cannot be unpacked."));
}

TEST_F(AggCoreTransformTest, ConfigureAndAttestRequestNoIntrinsicConfigs) {
  grpc::ClientContext context;
  Configuration invalid_config;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration must have exactly one IntrinsicConfig"));
}

TEST_F(AggCoreTransformTest,
       FedSqlDpGroupByInvalidParametersConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
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

  auto status = stub_->ConfigureAndAttest(&context, request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr(
          "`fedsql_dp_group_by` parameters must both be have type DT_DOUBLE"));
}
TEST_F(AggCoreTransformTest, MultipleTopLevelIntrinsicsConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
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

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Configuration must have exactly one IntrinsicConfig"));
}

TEST_F(AggCoreTransformTest, ConfigureAndAttestMoreThanOnce) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->ConfigureAndAttest(&context, request, &response).ok());

  grpc::ClientContext second_context;
  auto status = stub_->ConfigureAndAttest(&second_context, request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(AggCoreTransformTest, ValidConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_->ConfigureAndAttest(&context, request, &response).ok());
}

TEST_F(AggCoreTransformTest,
       FedSqlDpGroupByConfigureAndAttestGeneratesConfigProperties) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
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

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_TRUE(cwt.ok());
  ASSERT_EQ(cwt->config_properties.fields().at("intrinsic_uri").string_value(),
            "fedsql_dp_group_by");
  ASSERT_EQ(cwt->config_properties.fields().at("epsilon").number_value(), 1.1);
  ASSERT_EQ(cwt->config_properties.fields().at("delta").number_value(), 2.2);
}

TEST_F(AggCoreTransformTest, TransformZeroInputsReturnsInvalidArgumentError) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto status = stub_->Transform(&transform_context, transform_request,
                                 &transform_response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr(
          "The aggregation can't be successfully completed because no inputs "
          "were aggregated"));
}

TEST_F(AggCoreTransformTest, TransformExecutesFederatedSum) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt32TensorCheckpoint("foo", {1}));
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt32TensorCheckpoint("foo", {2}));
  for (auto& input : *transform_request.mutable_inputs()) {
    input.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  }
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 3);
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

TEST_F(AggCoreTransformTest, TransformExecutesFedSqlGroupBy) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  Configuration config = PARSE_TEXT_PROTO(R"pb(
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
  configure_request.mutable_configuration()->PackFrom(config);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  std::string checkpoint_1 = BuildFedSqlGroupByCheckpoint({1, 1, 2}, {1, 2, 5});
  std::string checkpoint_2 = BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0});

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(checkpoint_1);
  transform_request.add_inputs()->set_unencrypted_data(checkpoint_2);
  for (auto& input : *transform_request.mutable_inputs()) {
    input.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  }
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok()) << transform_status.error_message();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query sums the val column, grouping by key
  ASSERT_EQ(col_values->num_elements(), 3);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_THAT(col_values->AsSpan<int64_t>(), UnorderedElementsAre(7, 5, 0));
}

TEST_F(AggCoreTransformTest, MultipleTransformExecutionsSucceed) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt32TensorCheckpoint("foo", {1}));
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt32TensorCheckpoint("foo", {2}));
  for (auto& input : *transform_request.mutable_inputs()) {
    input.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  }
  grpc::ClientContext transform_context;
  TransformResponse transform_response;

  ASSERT_TRUE(stub_
                  ->Transform(&transform_context, transform_request,
                              &transform_response)
                  .ok());

  grpc::ClientContext transform_context_2;
  TransformResponse transform_response_2;
  ASSERT_TRUE(stub_
                  ->Transform(&transform_context_2, transform_request,
                              &transform_response_2)
                  .ok());
}

TEST_F(AggCoreTransformTest,
       TransformDecryptsMultipleRecordsAndExecutesFederatedSum) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "foo";
  std::string output_col_name = "foo_out";
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(3);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(3));

  std::string reencryption_public_key = "";
  std::string ciphertext_associated_data = "ciphertext associated data";

  std::string message_1 = BuildSingleInt32TensorCheckpoint(input_col_name, {1});
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(0), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  std::string message_2 = BuildSingleInt32TensorCheckpoint(input_col_name, {2});
  absl::StatusOr<Record> rewrapped_record_2 =
      crypto_test_utils::CreateRewrappedRecord(
          message_2, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(1), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_2.ok()) << rewrapped_record_2.status();

  std::string message_3 = BuildSingleInt32TensorCheckpoint(input_col_name, {3});
  absl::StatusOr<Record> rewrapped_record_3 =
      crypto_test_utils::CreateRewrappedRecord(
          message_3, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(2), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_3.ok()) << rewrapped_record_3.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_1);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_2);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_3);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 6);
}

TEST_F(AggCoreTransformTest, TransformIgnoresUndecryptableInputs) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "foo";
  std::string output_col_name = "foo_out";
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(3);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(3));

  std::string reencryption_public_key = "";
  std::string ciphertext_associated_data = "ciphertext associated data";

  std::string message_1 = BuildSingleInt32TensorCheckpoint(input_col_name, {1});
  // Create two records that will fail to decrypt and one record that can be
  // successfully decrypted.
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(0), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();
  rewrapped_record_1->mutable_hpke_plus_aead_data()->set_ciphertext("invalid");

  std::string message_2 = BuildSingleInt32TensorCheckpoint(input_col_name, {2});
  absl::StatusOr<Record> rewrapped_record_2 =
      crypto_test_utils::CreateRewrappedRecord(
          message_2, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(1), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_2.ok()) << rewrapped_record_2.status();
  rewrapped_record_2->mutable_hpke_plus_aead_data()->set_ciphertext(
      "undecryptable");

  std::string message_3 =
      BuildSingleInt32TensorCheckpoint(input_col_name, {42});
  absl::StatusOr<Record> rewrapped_record_3 =
      crypto_test_utils::CreateRewrappedRecord(
          message_3, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(2), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_3.ok()) << rewrapped_record_3.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_1);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_2);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_3);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());
  // Ensure that the number of ignored inputs is recorded.
  ASSERT_EQ(transform_response.num_ignored_inputs(), 2);

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 42);
}

TEST_F(AggCoreTransformTest, TransformIgnoresUnparseableInputs) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt32TensorCheckpoint("foo", {42}));
  transform_request.add_inputs()->set_unencrypted_data("invalid_checkpoint");
  transform_request.add_inputs()->set_unencrypted_data(
      "another_invalid_checkpoint");
  for (auto& input : *transform_request.mutable_inputs()) {
    input.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  }
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());
  // Ensure that the number of ignored inputs is recorded.
  ASSERT_EQ(transform_response.num_ignored_inputs(), 2);

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("foo_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT32);
  ASSERT_EQ(col_values->AsSpan<int32_t>().at(0), 42);
}

TEST_F(AggCoreTransformTest, TransformFailsIfInputCannotBeAccumulated) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  configure_request.mutable_configuration()->PackFrom(DefaultConfiguration());

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string message = BuildSingleInt32TensorCheckpoint("bad_col_name", {1});
  Record record;
  record.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  record.set_unencrypted_data(message);

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(record);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_EQ(transform_status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(AggCoreTransformTest, TransformBeforeConfigureAndAttest) {
  grpc::ClientContext context;
  TransformRequest request;
  TransformResponse response;
  auto status = stub_->Transform(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest must be called before Transform"));
}

TEST_F(AggCoreTransformTest, GenerateNoncesBeforeConfigureAndAttest) {
  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  auto status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr("ConfigureAndAttest must be called before GenerateNonces"));
}

}  // namespace

}  // namespace confidential_federated_compute::agg_core
