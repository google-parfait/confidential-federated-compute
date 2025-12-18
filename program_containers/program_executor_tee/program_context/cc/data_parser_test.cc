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
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "program_executor_tee/private_state.h"
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
using ::testing::Return;
using ::testing::Sequence;

template <typename T>
using Pair = typename tensorflow_federated::aggregation::AggVectorIterator<
    T>::IndexValuePair;

class MockPrivateState : public PrivateState {
 public:
  MockPrivateState()
      : PrivateState(/*initial_state=*/std::nullopt,
                     /*next_update_state=*/BudgetState{}) {}
  MOCK_METHOD(void, SetReleaseInitialState, (std::string initial_state),
              (override));
  MOCK_METHOD(std::optional<std::string>, GetReleaseInitialState, (),
              (override, const));
  MOCK_METHOD(std::string, GetReleaseUpdateState, (), (override));
};

class DataParserTest : public ::testing::Test {
 public:
  void SetUp() override {
    fake_data_read_write_service_ =
        std::make_unique<FakeDataReadWriteService>();

    const std::string localhost = "[::1]:";
    int data_read_write_service_port;
    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(localhost + "0",
                                             grpc::InsecureServerCredentials(),
                                             &data_read_write_service_port);
    data_read_write_builder.RegisterService(
        fake_data_read_write_service_.get());
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    data_read_write_server_address_ =
        localhost + std::to_string(data_read_write_service_port);

    mock_signing_key_handle_ =
        std::make_shared<NiceMock<MockSigningKeyHandle>>();
    google::protobuf::Struct config_properties;
    input_blob_decryptor_ =
        std::make_unique<confidential_federated_compute::BlobDecryptor>(
            *mock_signing_key_handle_, config_properties,
            std::vector<absl::string_view>(
                {fake_data_read_write_service_->GetInputPublicPrivateKeyPair()
                     .second}));

    data_parser_ = std::make_unique<DataParser>(
        input_blob_decryptor_.get(), data_read_write_server_address_,
        absl::Base64Escape(
            fake_data_read_write_service_->GetResultPublicPrivateKeyPair()
                .first),
        absl::Base64Escape(kAccessPolicyHash), &mock_private_state_,
        mock_signing_key_handle_,
        std::set<std::string>({absl::Base64Escape(kAccessPolicyHash)}));
  }

  void TearDown() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  std::string data_read_write_server_address_;
  std::unique_ptr<FakeDataReadWriteService> fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;

  MockPrivateState mock_private_state_;
  std::shared_ptr<NiceMock<MockSigningKeyHandle>> mock_signing_key_handle_;
  std::unique_ptr<BlobDecryptor> input_blob_decryptor_;
  std::unique_ptr<DataParser> data_parser_;
};

TEST_F(DataParserTest, ResolveUriToTensor_PlaintextIntCheckpoint) {
  std::string tensor_name = "tensor_name";
  std::string uri_1 = "test_uri_1";
  std::string uri_2 = "test_uri_2";
  CHECK_OK(this->fake_data_read_write_service_->StorePlaintextMessage(
      uri_1, BuildClientCheckpointFromInts({4, 5, 6}, tensor_name)));
  CHECK_OK(this->fake_data_read_write_service_->StorePlaintextMessage(
      uri_2, BuildClientCheckpointFromInts({7, 8, 9}, tensor_name)));

  // Resolve uri_1, then uri_2, then uri_1.
  auto tensor_proto = data_parser_->ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
  tensor_proto = data_parser_->ResolveUriToTensor(uri_2, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 7}, Pair<int>{1, 8}, Pair<int>{2, 9}));
  tensor_proto = data_parser_->ResolveUriToTensor(uri_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));

  std::vector<std::string> requested_uris =
      fake_data_read_write_service_->GetReadRequestUris();
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
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      uri, checkpoint));
  auto tensor_proto = data_parser_->ResolveUriToTensor(uri, tensor_name);
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
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      uri, checkpoint));
  auto tensor_proto = data_parser_->ResolveUriToTensor(uri, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<absl::string_view>(),
              ElementsAre(Pair<absl::string_view>{0, "serialized_example_1"},
                          Pair<absl::string_view>{1, "serialized_example_2"}));
}

TEST_F(DataParserTest, ResolveUriToTensor_IncorrectCheckpointFormat) {
  std::string message = "not a fc checkpoint";
  std::string uri = "test_uri";
  CHECK_OK(
      fake_data_read_write_service_->StoreEncryptedMessageForKms(uri, message));
  auto tensor_proto =
      data_parser_->ResolveUriToTensor(uri, "unused_tensor_name");
  EXPECT_EQ(tensor_proto.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(tensor_proto.status().message(),
              HasSubstr("Unsupported checkpoint format"));
}

TEST_F(DataParserTest, ResolveUriToTensor_IncorrectTensorName) {
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, "tensor_name");
  std::string uri = "test_uri";
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      uri, checkpoint));
  auto tensor_proto =
      data_parser_->ResolveUriToTensor(uri, "different_tensor_name");
  EXPECT_EQ(tensor_proto.status(),
            absl::NotFoundError(
                "No aggregation tensor found for name different_tensor_name"));
}

TEST_F(DataParserTest, ResolveUriToTensor_RepeatedBlobId) {
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, tensor_name);
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      "uri_1", checkpoint, "blob_id_1"));
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      "uri_2", checkpoint, "blob_id_2"));
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      "uri_3", checkpoint, "blob_id_1"));

  // There should be no issue with receiving blob_id_1 for a lookup for uri_1.
  auto tensor_proto = data_parser_->ResolveUriToTensor("uri_1", tensor_name);
  ASSERT_TRUE(tensor_proto.ok());

  // There should be no issue with receiving blob_id_2 for a lookup for uri_2.
  tensor_proto = data_parser_->ResolveUriToTensor("uri_2", tensor_name);
  ASSERT_TRUE(tensor_proto.ok());

  // Receiving blob_id_1 for a lookup for uri_3 should fail because blob_id_1
  // was previously seen for uri_1.
  tensor_proto = data_parser_->ResolveUriToTensor("uri_3", tensor_name);
  EXPECT_EQ(
      tensor_proto.status(),
      absl::InvalidArgumentError(
          "This blob id was previously returned for a different filename."));
}

TEST_F(DataParserTest, ReleaseUnencrypted) {
  std::string kStateAfterFirstRelease = "state_after_first_release";
  std::string kStateAfterSecondRelease = "state_after_second_release";

  Sequence s;
  EXPECT_CALL(mock_private_state_, GetReleaseUpdateState())
      .InSequence(s)
      .WillOnce(Return(kStateAfterFirstRelease));
  EXPECT_CALL(mock_private_state_, GetReleaseInitialState())
      .InSequence(s)
      .WillOnce(Return(std::nullopt));
  EXPECT_CALL(mock_private_state_,
              SetReleaseInitialState(kStateAfterFirstRelease))
      .InSequence(s);
  EXPECT_CALL(mock_private_state_, GetReleaseUpdateState())
      .InSequence(s)
      .WillOnce(Return(kStateAfterSecondRelease));
  EXPECT_CALL(mock_private_state_, GetReleaseInitialState())
      .InSequence(s)
      .WillOnce(Return(kStateAfterFirstRelease));
  EXPECT_CALL(mock_private_state_,
              SetReleaseInitialState(kStateAfterSecondRelease))
      .InSequence(s);

  ASSERT_TRUE(data_parser_->ReleaseUnencrypted("abc", "my_key_1").ok());
  ASSERT_TRUE(data_parser_->ReleaseUnencrypted("def", "my_key_2").ok());

  auto released_data = fake_data_read_write_service_->GetReleasedData();
  EXPECT_EQ(released_data.size(), 2);
  EXPECT_EQ(released_data["my_key_1"], "abc");
  EXPECT_EQ(released_data["my_key_2"], "def");

  std::map<std::string, std::pair<std::optional<std::optional<std::string>>,
                                  std::optional<std::string>>>
      released_state_changes =
          fake_data_read_write_service_->GetReleasedStateChanges();
  ASSERT_EQ(released_state_changes.size(), 2);
  auto state_change_1 = released_state_changes["my_key_1"];
  ASSERT_EQ(state_change_1.first.value(), std::nullopt);
  ASSERT_EQ(state_change_1.second.value(), kStateAfterFirstRelease);
  auto state_change_2 = released_state_changes["my_key_2"];
  ASSERT_EQ(state_change_2.first.value().value(), kStateAfterFirstRelease);
  ASSERT_EQ(state_change_2.second.value(), kStateAfterSecondRelease);
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee