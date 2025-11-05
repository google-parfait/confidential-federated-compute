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

#include "program_executor_tee/program_context/cc/data_parser.h"

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_iterator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;

template <typename T>
using Pair = typename tensorflow_federated::aggregation::AggVectorIterator<
    T>::IndexValuePair;

class DataParserTest : public ::testing::Test {
 public:
  DataParserTest() {
    const std::string localhost = "[::1]:";
    int data_read_write_service_port;
    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(localhost + "0",
                                             grpc::InsecureServerCredentials(),
                                             &data_read_write_service_port);
    data_read_write_builder.RegisterService(&fake_data_read_write_service_);
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    data_read_write_server_address_ =
        localhost + std::to_string(data_read_write_service_port);
  }

  ~DataParserTest() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  std::string data_read_write_server_address_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
};

TEST_F(DataParserTest, ResolveUriToTensor_PlaintextIntCheckpoint) {
  std::string tensor_name = "tensor_name";
  std::string uri = "test_uri";
  CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
      uri, BuildClientCheckpointFromInts({4, 5, 6}, tensor_name)));

  DataParser data_parser(/*blob_decryptor=*/nullptr,
                         data_read_write_server_address_);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
}

TEST_F(DataParserTest, ResolveUriToTensor_PlaintextIntCheckpointWithCaching) {
  std::string tensor_name = "tensor_name";
  std::string uri_1 = "test_uri_1";
  std::string uri_2 = "test_uri_2";
  CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
      uri_1, BuildClientCheckpointFromInts({4, 5, 6}, tensor_name)));
  CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
      uri_2, BuildClientCheckpointFromInts({7, 8, 9}, tensor_name)));

  // Resolve uri_1, then uri_2, then uri_1.
  DataParser data_parser(/*blob_decryptor=*/nullptr,
                         data_read_write_server_address_);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
  tensor_proto = data_parser.ResolveUriToTensor(uri_2, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 7}, Pair<int>{1, 8}, Pair<int>{2, 9}));
  tensor_proto = data_parser.ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));

  // Due to caching, the FakeDataReadWriteService should have only recorded
  // requests for uri_1 followed by uri_2.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 2);
  EXPECT_EQ(requested_uris[0], uri_1);
  EXPECT_EQ(requested_uris[1], uri_2);
}

TEST_F(DataParserTest,
       ResolveUriToTensor_PlaintextIntCheckpointWithoutCaching) {
  std::string tensor_name = "tensor_name";
  std::string uri_1 = "test_uri_1";
  std::string uri_2 = "test_uri_2";
  CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
      uri_1, BuildClientCheckpointFromInts({4, 5, 6}, tensor_name)));
  CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
      uri_2, BuildClientCheckpointFromInts({7, 8, 9}, tensor_name)));

  // Resolve uri_1, then uri_2, then uri_1.
  DataParser data_parser(/*blob_decryptor=*/nullptr,
                         data_read_write_server_address_,
                         /*use_caching=*/false);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
  tensor_proto = data_parser.ResolveUriToTensor(uri_2, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 7}, Pair<int>{1, 8}, Pair<int>{2, 9}));
  tensor_proto = data_parser.ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));

  // The FakeDataReadWriteService should have recorded uri_1, then uri_2, then
  // uri_1.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 3);
  EXPECT_EQ(requested_uris[0], uri_1);
  EXPECT_EQ(requested_uris[1], uri_2);
  EXPECT_EQ(requested_uris[2], uri_1);
}

TEST_F(DataParserTest, ResolveUriToTensor_EncryptedIntCheckpoint) {
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, tensor_name);
  std::string uri = "test_uri";
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, checkpoint, "ciphertext associated data", *recipient_public_key,
      nonce, "reencryption_public_key"));

  std::function<std::string()> nonce_generator = [&]() { return nonce; };
  DataParser data_parser(&blob_decryptor, data_read_write_server_address_,
                         /*use_caching=*/true, nonce_generator);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
}

TEST_F(DataParserTest, ResolveUriToTensor_EncryptedStringCheckpoint) {
  std::string tensor_name = "tensor_name";
  std::string checkpoint = BuildClientCheckpointFromStrings(
      {"serialized_example_1", "serialized_example_2"}, tensor_name);
  std::string uri = "test_uri";
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, checkpoint, "ciphertext associated data", *recipient_public_key,
      nonce, "reencryption_public_key"));

  std::function<std::string()> nonce_generator = [&]() { return nonce; };
  DataParser data_parser(&blob_decryptor, data_read_write_server_address_,
                         /*use_caching=*/true, nonce_generator);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<absl::string_view>(),
              ElementsAre(Pair<absl::string_view>{0, "serialized_example_1"},
                          Pair<absl::string_view>{1, "serialized_example_2"}));
}

TEST_F(DataParserTest, ResolveUriToTensor_MismatchedNonce) {
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, tensor_name);
  std::string uri = "test_uri";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, checkpoint, "ciphertext associated data", *recipient_public_key,
      "nonce", "reencryption_public_key"));

  std::function<std::string()> nonce_generator = [&]() {
    return "different nonce";
  };
  DataParser data_parser(&blob_decryptor, data_read_write_server_address_,
                         /*use_caching=*/true, nonce_generator);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri, tensor_name);
  EXPECT_EQ(tensor_proto.status(),
            absl::InvalidArgumentError("ReadResponse nonce does not match"));
}

TEST_F(DataParserTest, ResolveUriToTensor_IncorrectCheckpointFormat) {
  std::string message = "not a fc checkpoint";
  std::string uri = "test_uri";
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

  std::function<std::string()> nonce_generator = [&]() { return nonce; };
  DataParser data_parser(&blob_decryptor, data_read_write_server_address_,
                         /*use_caching=*/true, nonce_generator);
  auto tensor_proto = data_parser.ResolveUriToTensor(uri, "unused_tensor_name");
  EXPECT_EQ(tensor_proto.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(tensor_proto.status().message(),
              HasSubstr("Unsupported checkpoint format"));
}

TEST_F(DataParserTest, ResolveUriToTensor_IncorrectTensorName) {
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, "tensor_name");
  std::string uri = "test_uri";
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, checkpoint, "ciphertext associated data", *recipient_public_key,
      nonce, "reencryption_public_key"));

  std::function<std::string()> nonce_generator = [&]() { return nonce; };
  DataParser data_parser(&blob_decryptor, data_read_write_server_address_,
                         /*use_caching=*/true, nonce_generator);
  auto tensor_proto =
      data_parser.ResolveUriToTensor(uri, "different_tensor_name");
  EXPECT_EQ(tensor_proto.status(),
            absl::NotFoundError(
                "No aggregation tensor found for name different_tensor_name"));
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee