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

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

template <typename T>
using Pair = typename tensorflow_federated::aggregation::AggVectorIterator<
    T>::IndexValuePair;

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

    recovery_info_public_private_key_pair_ =
        crypto_test_utils::GenerateKeyPair("recovery");
    std::string input_private_key =
        fake_data_read_write_service_->GetInputPublicPrivateKeyPair().second;
    auto decryption_keys = std::vector<absl::string_view>(
        {input_private_key, recovery_info_public_private_key_pair_.second});
    reencryption_keys_ = std::vector<std::string>{
        absl::Base64Escape(recovery_info_public_private_key_pair_.first),
        absl::Base64Escape(
            fake_data_read_write_service_->GetResultPublicPrivateKeyPair()
                .first)};

    input_blob_decryptor_ =
        std::make_unique<confidential_federated_compute::Decryptor>(
            std::move(decryption_keys));
  }

  // Create a DataParser with the given initial PrivateState.
  void InitDataParser(std::optional<std::string> initial_state = std::nullopt) {
    auto private_state_or =
        PrivateState::CreatePrivateState(std::move(initial_state));
    CHECK_OK(private_state_or);
    private_state_ = std::move(*private_state_or);

    data_parser_ = std::make_unique<DataParser>(
        input_blob_decryptor_.get(), data_read_write_server_address_,
        reencryption_keys_, absl::Base64Escape(kAccessPolicyHash),
        absl::Base64Escape(fake_data_read_write_service_->GetKmsPublicKey()),
        absl::Base64Escape(kTestInvocationId), private_state_.get(),
        std::shared_ptr<oak::crypto::SigningKeyHandle>(
            fake_data_read_write_service_->GetOakSigningKeyHandle()),
        std::set<std::string>({absl::Base64Escape(kAccessPolicyHash)}));
  }

  void TearDown() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  std::string data_read_write_server_address_;
  std::unique_ptr<FakeDataReadWriteService> fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;

  std::unique_ptr<PrivateState> private_state_;
  std::unique_ptr<DataParser> data_parser_;
  std::unique_ptr<confidential_federated_compute::Decryptor>
      input_blob_decryptor_;

  std::vector<std::string> reencryption_keys_;
  std::pair<std::string, std::string> recovery_info_public_private_key_pair_;
};

TEST_F(DataParserTest, ResolveBlobIdToTensor_PlaintextIntCheckpoint) {
  InitDataParser();
  std::string tensor_name = "tensor_name";
  std::string blob_id_1 = "test_blob_id_1";
  std::string blob_id_2 = "test_blob_id_2";
  CHECK_OK(this->fake_data_read_write_service_->StorePlaintextMessage(
      blob_id_1, BuildClientCheckpointFromInts({4, 5, 6}, tensor_name)));
  CHECK_OK(this->fake_data_read_write_service_->StorePlaintextMessage(
      blob_id_2, BuildClientCheckpointFromInts({7, 8, 9}, tensor_name)));

  // Resolve blob_id_1, then blob_id_2, then blob_id_1.
  auto tensor_proto =
      data_parser_->ResolveBlobIdToTensor(blob_id_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
  tensor_proto = data_parser_->ResolveBlobIdToTensor(blob_id_2, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 7}, Pair<int>{1, 8}, Pair<int>{2, 9}));
  tensor_proto = data_parser_->ResolveBlobIdToTensor(blob_id_1, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));

  std::vector<std::string> requested_ids =
      fake_data_read_write_service_->GetReadRequestIds();
  EXPECT_EQ(requested_ids.size(), 3);
  EXPECT_EQ(requested_ids[0], blob_id_1);
  EXPECT_EQ(requested_ids[1], blob_id_2);
  EXPECT_EQ(requested_ids[2], blob_id_1);
}

TEST_F(DataParserTest, ResolveBlobIdToTensor_EncryptedIntCheckpoint) {
  InitDataParser();
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, tensor_name);
  std::string blob_id = "test_blob_id";
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      blob_id, checkpoint));
  auto tensor_proto = data_parser_->ResolveBlobIdToTensor(blob_id, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<int>(),
              ElementsAre(Pair<int>{0, 4}, Pair<int>{1, 5}, Pair<int>{2, 6}));
}

TEST_F(DataParserTest, ResolveBlobIdToTensor_EncryptedStringCheckpoint) {
  InitDataParser();
  std::string tensor_name = "tensor_name";
  std::string checkpoint = BuildClientCheckpointFromStrings(
      {"serialized_example_1", "serialized_example_2"}, tensor_name);
  std::string blob_id = "test_blob_id";
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      blob_id, checkpoint));
  auto tensor_proto = data_parser_->ResolveBlobIdToTensor(blob_id, tensor_name);
  ASSERT_TRUE(tensor_proto.ok());
  auto tensor = Tensor::FromProto(std::move(*tensor_proto));
  ASSERT_TRUE(tensor.ok());
  EXPECT_THAT(tensor->AsAggVector<absl::string_view>(),
              ElementsAre(Pair<absl::string_view>{0, "serialized_example_1"},
                          Pair<absl::string_view>{1, "serialized_example_2"}));
}

TEST_F(DataParserTest, ResolveBlobIdToTensor_IncorrectCheckpointFormat) {
  InitDataParser();
  std::string message = "not a fc checkpoint";
  std::string blob_id = "test_blob_id";
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(blob_id,
                                                                      message));
  auto tensor_proto =
      data_parser_->ResolveBlobIdToTensor(blob_id, "unused_tensor_name");
  EXPECT_EQ(tensor_proto.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(tensor_proto.status().message(),
              HasSubstr("Unsupported checkpoint format"));
}

TEST_F(DataParserTest, ResolveBlobIdToTensor_IncorrectTensorName) {
  InitDataParser();
  std::string checkpoint =
      BuildClientCheckpointFromInts({4, 5, 6}, "tensor_name");
  std::string blob_id = "test_blob_id";
  CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
      blob_id, checkpoint));
  auto tensor_proto =
      data_parser_->ResolveBlobIdToTensor(blob_id, "different_tensor_name");
  EXPECT_EQ(tensor_proto.status(),
            absl::NotFoundError(
                "No aggregation tensor found for name different_tensor_name"));
}

TEST_F(DataParserTest, ReleaseUnencrypted) {
  InitDataParser();

  ASSERT_TRUE(data_parser_->ReleaseUnencrypted("abc", "my_key_1").ok());
  ASSERT_TRUE(data_parser_->ReleaseUnencrypted("def", "my_key_2").ok());

  auto released_data = fake_data_read_write_service_->GetReleasedData();
  EXPECT_EQ(released_data.size(), 2);
  EXPECT_EQ(released_data["my_key_1"], "abc");
  EXPECT_EQ(released_data["my_key_2"], "def");

  // Verify the state transitions: the first release should have no initial
  // state and a counter of 1, the second should use the first's state as its
  // initial state and have a counter of 2.
  std::map<std::string, std::pair<std::optional<std::optional<std::string>>,
                                  std::optional<std::string>>>
      released_state_changes =
          fake_data_read_write_service_->GetReleasedStateChanges();
  ASSERT_EQ(released_state_changes.size(), 2);

  auto state_change_1 = released_state_changes["my_key_1"];
  ASSERT_EQ(state_change_1.first.value(), std::nullopt);
  BudgetState dst_state_1;
  ASSERT_TRUE(dst_state_1.ParseFromString(state_change_1.second.value()));
  EXPECT_EQ(dst_state_1.counter(), 1);

  auto state_change_2 = released_state_changes["my_key_2"];
  BudgetState src_state_2;
  ASSERT_TRUE(
      src_state_2.ParseFromString(state_change_2.first.value().value()));
  EXPECT_EQ(src_state_2.counter(), 1);
  BudgetState dst_state_2;
  ASSERT_TRUE(dst_state_2.ParseFromString(state_change_2.second.value()));
  EXPECT_EQ(dst_state_2.counter(), 2);
}

TEST_F(DataParserTest, ReleaseUnencrypted_UnsupportedAfterSaveRecovery) {
  InitDataParser();

  // Save recovery info first. This sets HasPriorSaveRecovery to true.
  ASSERT_TRUE(
      data_parser_->SaveRecoveryInfo("recovery_info_value", "recovery_key", {})
          .ok());

  // Now calling ReleaseUnencrypted should fail.
  auto status = data_parser_->ReleaseUnencrypted("abc", "my_key_1");
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Releasing unencrypted information without associated recovery"));
}

TEST_F(DataParserTest, SaveAndRestoreRecoveryInfo) {
  InitDataParser();

  ASSERT_TRUE(data_parser_
                  ->SaveRecoveryInfo("recovery_info_value", "recovery_key",
                                     {{"abc", "my_key_1"}, {"def", "my_key_2"}})
                  .ok());

  auto recovery_info = data_parser_->RestoreRecoveryInfo("recovery_key");
  ASSERT_TRUE(recovery_info.ok()) << recovery_info.status().ToString();
  EXPECT_EQ(recovery_info.value(), "recovery_info_value");

  // Check that the unencrypted data was also released.
  auto released_data = fake_data_read_write_service_->GetReleasedData();
  EXPECT_EQ(released_data.size(), 2);
  EXPECT_EQ(released_data["my_key_1"], "abc");
  EXPECT_EQ(released_data["my_key_2"], "def");
}

TEST_F(DataParserTest, RestoreRecoveryInfo_FailStale) {
  // Start with no initial state. SaveRecoveryInfo twice: the first save
  // includes a release to commit state with the first recovery blob id, and
  // the second save updates the recovery blob id again. Restoring the first
  // recovery info should fail because AllowRestoreRecovery will reject the
  // stale blob id.
  InitDataParser();

  // Save first recovery info, including a release to commit state.
  ASSERT_TRUE(
      data_parser_->SaveRecoveryInfo("first_value", "recovery_key_1", {{}})
          .ok());

  // Save second recovery info, which updates the recovery blob id again.
  ASSERT_TRUE(
      data_parser_->SaveRecoveryInfo("second_value", "recovery_key_2", {})
          .ok());

  // Restoring the first recovery info should fail: the private state's
  // recovery_blob_id was updated to the second save's blob id, which won't
  // match the first save's blob id or its committed_blob_id.
  auto recovery_info = data_parser_->RestoreRecoveryInfo("recovery_key_1");
  ASSERT_FALSE(recovery_info.ok());
  EXPECT_EQ(recovery_info.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(recovery_info.status().message(),
              HasSubstr("does not match the expected state for recovery"));
}

TEST_F(DataParserTest, SaveRecoveryInfo_FailsWhenRestoreExpectedFirst) {
  // First, save recovery info with a release so that the state gets committed
  // with a recovery_blob_id.
  InitDataParser();
  ASSERT_TRUE(data_parser_
                  ->SaveRecoveryInfo("recovery_value", "recovery_key",
                                     {{"data", "release_key"}})
                  .ok());

  // Re-initialize with the committed state, which has a recovery_blob_id.
  // This simulates a new program execution that must restore before saving.
  std::string committed_state = private_state_->GetState().value();
  InitDataParser(committed_state);

  // SaveRecoveryInfo should fail because AllowSaveRecovery returns false
  // until recovery is restored first.
  auto status =
      data_parser_->SaveRecoveryInfo("new_value", "new_recovery_key", {});
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.message(),
      HasSubstr("Saving recovery information is unsupported if a previous "
                "program execution previously released information but the "
                "corresponding recovery information has not yet been loaded."));

  // After restoring the recovery info, SaveRecoveryInfo should succeed.
  auto recovery_info = data_parser_->RestoreRecoveryInfo("recovery_key");
  ASSERT_TRUE(recovery_info.ok()) << recovery_info.status().ToString();
  EXPECT_EQ(recovery_info.value(), "recovery_value");

  ASSERT_TRUE(
      data_parser_->SaveRecoveryInfo("new_value", "new_recovery_key", {}).ok());
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee