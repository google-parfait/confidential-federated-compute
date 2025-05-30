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

#include "containers/program_executor_tee/program_context/cc/data_uri_resolver.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "fcp/protos/confidentialcompute/file_info.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "testing/matchers.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Test;

// Default name of key in federated compute checkpoint where client examples
// will be stored.
inline constexpr char kDataTensorName[] = "client_examples";

// Construct a federated compute checkpoint consisting of a one-dimensional
// string tensor of inputs stored at a given key.
std::string BuildClientCheckpointFromStrings(
    std::vector<std::string> input_values, std::string key_name) {
  auto data = std::make_unique<MutableStringData>(input_values.size());
  for (std::string value : input_values) {
    data->Add(std::move(value));
  }
  absl::StatusOr<Tensor> t =
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     std::move(data));
  CHECK_OK(t);
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();
  CHECK_OK(ckpt_builder->Add(key_name, *t));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint);
  return std::string(*checkpoint);
}

class DataUriResolverTest : public Test {
 public:
  DataUriResolverTest() {
    const std::string server_address = "[::1]:";

    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(
        server_address + "0", grpc::InsecureServerCredentials(), &port_);
    data_read_write_builder.RegisterService(&fake_data_read_write_service_);
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    LOG(INFO) << "DataReadWrite server listening on "
              << server_address + std::to_string(port_) << std::endl;
  }

  ~DataUriResolverTest() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  int port_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
};

TEST_F(DataUriResolverTest, ResolveToValueSucceeds) {
  // Prepopulate the FakeDataReadWriteService with data.
  std::string uri = "my_uri";
  std::vector<std::string> examples = {"example1", "example2", "example3"};
  std::string message =
      BuildClientCheckpointFromStrings(examples, std::string(kDataTensorName));
  std::string nonce = "nonce";
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

  // Construct the DataUriResolver.
  std::function<std::string()> nonce_generator = [&nonce]() { return nonce; };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  // Create a data pointer corresponding to the stored data.
  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri(uri);
  file_info.set_key(kDataTensorName);
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Use the DataUriResolver to lookup the stored data.
  federated_language::Type federated_type;
  federated_language::TensorType* tensor_type = federated_type.mutable_tensor();
  tensor_type->set_dtype(federated_language::DataType::DT_STRING);
  tensor_type->add_dims(examples.size());
  tensorflow_federated::v0::Value value_out;
  EXPECT_OK(resolver.ResolveToValue(data_value, federated_type, value_out));

  // Check that the DataReadWrite service was called for the correct uri.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);

  // Check that the returned tensor holds the original examples.
  tensorflow::Tensor output_tensor =
      tensorflow_federated::DeserializeTensorValue(value_out).value();
  EXPECT_EQ(output_tensor.NumElements(), 3);
  auto flat_tensor = output_tensor.flat<tensorflow::tstring>();
  EXPECT_EQ(flat_tensor(0), examples[0]) << flat_tensor(0);
  EXPECT_EQ(flat_tensor(1), examples[1]) << flat_tensor(1);
  EXPECT_EQ(flat_tensor(2), examples[2]) << flat_tensor(2);
  EXPECT_EQ(value_out.value_case(),
            tensorflow_federated::v0::Value::ValueCase::kArray);
  EXPECT_EQ(value_out.array().dtype(), federated_language::DataType::DT_STRING);
  EXPECT_EQ(value_out.array().shape().dim().size(), 1);
  EXPECT_EQ(value_out.array().shape().dim(0), examples.size());

  // Resolve the same uri a second time and check that it did not result in a
  // second call to the DataReadWrite service.
  tensorflow_federated::v0::Value value_out_again;
  EXPECT_OK(
      resolver.ResolveToValue(data_value, federated_type, value_out_again));
  std::vector<std::string> requested_uris_again =
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris_again.size(), 1);
  EXPECT_EQ(requested_uris_again[0], uri);
  tensorflow::Tensor output_tensor_again =
      tensorflow_federated::DeserializeTensorValue(value_out_again).value();
  EXPECT_EQ(output_tensor_again.NumElements(), 3);
}

TEST_F(DataUriResolverTest, ResolveToValueMalformedDataProto) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  std::function<std::string()> nonce_generator = []() { return "nonce"; };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  federated_language::Data data_value;
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr("Expected Data content field to contain FileInfo proto"));
}

TEST_F(DataUriResolverTest, ResolveToValueDataReadWriteServiceFailure) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  std::function<std::string()> nonce_generator = []() { return "nonce"; };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri("my_uri");
  file_info.set_key(kDataTensorName);
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Do not pre-populate the FakeDataReadWriteService with any data, causing
  // all ReadRequests to fail.
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  ASSERT_THAT(status.message(), HasSubstr("Error receiving ReadResponse"));
}

TEST_F(DataUriResolverTest, ResolveToValueMismatchingNonceFailure) {
  // Prepopulate the FakeDataReadWriteService with data.
  std::string uri = "my_uri";
  std::vector<std::string> examples = {"example1", "example2", "example3"};
  std::string message =
      BuildClientCheckpointFromStrings(examples, std::string(kDataTensorName));
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key,
      "nonce", "reencryption_public_key"));

  // Construct the DataUriResolver using a nonce generator that will generate
  // a mismatching nonce.
  std::function<std::string()> nonce_generator = []() {
    return "different_nonce";
  };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  // Create a data pointer corresponding to the stored data.
  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri(uri);
  file_info.set_key(kDataTensorName);
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Use the DataUriResolver to lookup the stored data.
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  ASSERT_THAT(status.message(), HasSubstr("Mismatched nonce"));
}

TEST_F(DataUriResolverTest, ResolveToValueDecryptionFailure) {
  // Prepopulate the FakeDataReadWriteService with data.
  std::string uri = "my_uri";
  std::vector<std::string> examples = {"example1", "example2", "example3"};
  std::string message =
      BuildClientCheckpointFromStrings(examples, std::string(kDataTensorName));
  std::string nonce = "nonce";
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

  // Construct the DataUriResolver using a BlobDecryptor whose public key will
  // not match the one used above to prepopulate the FakeDataReadWriteService.
  NiceMock<MockOrchestratorCryptoStub> second_mock_crypto_stub;
  BlobDecryptor second_blob_decryptor(second_mock_crypto_stub);
  std::function<std::string()> nonce_generator = [&nonce]() { return nonce; };
  DataUriResolver resolver(&second_blob_decryptor, nonce_generator, port_);

  // Create a data pointer corresponding to the stored data.
  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri(uri);
  file_info.set_key(kDataTensorName);
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Use the DataUriResolver to lookup the stored data.
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("Failed to unwrap symmetric key"));
}

TEST_F(DataUriResolverTest, ResolveToValueMalformedCheckpointFailure) {
  // Prepopulate the FakeDataReadWriteService with a malformed checkpoint.
  std::string uri = "my_uri";
  std::string message = "malformed checkpoint";
  std::string nonce = "nonce";
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

  // Construct the DataUriResolver.
  std::function<std::string()> nonce_generator = [&nonce]() { return nonce; };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  // Create a data pointer corresponding to the stored data.
  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri(uri);
  file_info.set_key(kDataTensorName);
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Use the DataUriResolver to lookup the stored data.
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(), HasSubstr("Unsupported checkpoint format"));
}

TEST_F(DataUriResolverTest, ResolveToValueMismatchedCheckpointKeyFailure) {
  // Prepopulate the FakeDataReadWriteService with data.
  std::string uri = "my_uri";
  std::vector<std::string> examples = {"example1", "example2", "example3"};
  std::string message =
      BuildClientCheckpointFromStrings(examples, std::string(kDataTensorName));
  std::string nonce = "nonce";
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

  // Construct the DataUriResolver.
  std::function<std::string()> nonce_generator = [&nonce]() { return nonce; };
  DataUriResolver resolver(&blob_decryptor, nonce_generator, port_);

  // Create a data pointer with a uri that matches the stored data but a key
  // that does not.
  fcp::confidentialcompute::FileInfo file_info;
  file_info.set_uri(uri);
  file_info.set_key("mismatched key");
  federated_language::Data data_value;
  data_value.mutable_content()->PackFrom(file_info);

  // Use the DataUriResolver to lookup the stored data.
  federated_language::Type federated_type;
  tensorflow_federated::v0::Value value_out;
  absl::Status status =
      resolver.ResolveToValue(data_value, federated_type, value_out);
  ASSERT_EQ(status.code(), absl::StatusCode::kNotFound);
  ASSERT_THAT(status.message(),
              HasSubstr("No aggregation tensor found for name"));
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee