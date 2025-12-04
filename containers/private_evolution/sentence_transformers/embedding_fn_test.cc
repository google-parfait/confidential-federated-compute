// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "embedding_fn.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/fns/map_fn.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "tensor_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {
constexpr absl::string_view kKeyId = "key_id";
constexpr absl::string_view kBlobId = "blob_id";
constexpr absl::string_view kConfigId = "config_id";
constexpr absl::string_view kEncryptedSymmetricKey = "encrypted_key";
constexpr absl::string_view kEncapsulatedPublicKey = "encapped_key";
constexpr absl::string_view kReencryptionPolicyHash = "policy_hash";

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::bazel::tools::cpp::runfiles::Runfiles;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::KeyValue;
using ::confidential_federated_compute::fns::MapFn;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::
    SentenceTransformersContainerInitializeConfiguration;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_STRING;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Return;
using ::testing::SaveArg;

class MockDelegate : public ModelDelegate {
 public:
  MOCK_METHOD(void, InitializeRuntime, (), (override));
  MOCK_METHOD(void, FinalizeRuntime, (), (override));
  MOCK_METHOD(bool, InitializeModel, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::vector<float>>>,
              GenerateEmbeddings,
              (const std::vector<std::string>&, std::optional<std::string>),
              (override));
};

class MockModelDelegateFactory : public ModelDelegateFactory {
 public:
  MOCK_METHOD(std::unique_ptr<ModelDelegate>, Create, (), (override));
};

class MockContext : public confidential_federated_compute::Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
  MOCK_METHOD(bool, EmitUnencrypted, (Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV), (override));
};

class EmbeddingMapFnTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
    ASSERT_NE(runfiles, nullptr) << error;
    std::string archive_path = runfiles->Rlocation("_main/test/my_model.zip");
    write_configuration_map_[std::string(kConfigId)] = archive_path;

    std::filesystem::path temp_base = std::filesystem::temp_directory_path();
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int random_number = std::rand();
    temp_dir_ = temp_base / std::to_string(random_number);
    if (!std::filesystem::exists(temp_dir_)) {
      ASSERT_TRUE(std::filesystem::create_directory(temp_dir_));
    }

    SentenceTransformersContainerInitializeConfiguration config;
    config.set_model_artifacts_configuration_id(kConfigId);
    init_config_.PackFrom(config);
  }

  void TearDown() override { std::filesystem::remove_all(temp_dir_); }

  std::unique_ptr<Fn> InitializeFn(
      std::optional<std::string> prompt = std::nullopt) {
    std::string temp_dir = std::string(temp_dir_);
    EmbeddingFnFactoryProvider provider =
        CreateEmbeddingFnFactoryProvider(factory_, temp_dir);

    if (prompt.has_value()) {
      SentenceTransformersContainerInitializeConfiguration config;
      init_config_.UnpackTo(&config);
      config.mutable_encode_config()->set_prompt(*prompt);
      init_config_.PackFrom(config);
    }
    auto map_fn_factory =
        provider(init_config_, Any(), write_configuration_map_);
    CHECK_OK(map_fn_factory);

    auto delegate = std::make_unique<MockDelegate>();
    delegate_raw_ptr_ = delegate.get();
    EXPECT_CALL(factory_, Create()).WillOnce(Return(std::move(delegate)));
    auto map_fn = (*map_fn_factory)->CreateFn();
    CHECK_OK(map_fn);
    return std::move(*map_fn);
  }

  MockContext context_;
  MockModelDelegateFactory factory_;
  WriteConfigurationMap write_configuration_map_;
  std::filesystem::path temp_dir_;
  Any init_config_;
  MockDelegate* delegate_raw_ptr_;
};

absl::StatusOr<std::string> CreateInputCheckpoint(
    std::vector<std::string> input) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  Tensor t(std::move(input));
  FCP_RETURN_IF_ERROR(builder->Add(std::string(kDataTensorName), t));
  FCP_ASSIGN_OR_RETURN(absl::Cord ckpt, builder->Build());
  std::string ckpt_str;
  absl::CopyCordToString(ckpt, &ckpt_str);
  return ckpt_str;
}

absl::StatusOr<std::string> CreateOutputCheckpoint(
    std::vector<std::vector<float>> output) {
  FCP_ASSIGN_OR_RETURN(auto t, CreateEmbeddingTensor(std::move(output)));
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  FCP_RETURN_IF_ERROR(builder->Add(std::string(kDataTensorName), std::move(t)));
  FCP_ASSIGN_OR_RETURN(absl::Cord ckpt, builder->Build());
  return std::string(std::move(ckpt));
}

BlobMetadata CreateBlobMetadata(const BlobHeader& header) {
  BlobMetadata blob_metadata;
  auto encryption_metadata = blob_metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->set_blob_id(header.blob_id());
  *encryption_metadata->mutable_ciphertext_associated_data() =
      header.SerializeAsString();
  *encryption_metadata->mutable_kms_symmetric_key_associated_data()
       ->mutable_record_header() = header.SerializeAsString();
  encryption_metadata->set_encrypted_symmetric_key(
      std::string(kEncryptedSymmetricKey));
  encryption_metadata->set_encapsulated_public_key(
      std::string(kEncapsulatedPublicKey));
  return blob_metadata;
}

TEST_F(EmbeddingMapFnTest, InvalidInitConfig) {
  std::string temp_dir = std::string(temp_dir_);
  EmbeddingFnFactoryProvider provider =
      CreateEmbeddingFnFactoryProvider(factory_, temp_dir);

  EXPECT_THAT(provider(Any(), Any(), write_configuration_map_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(EmbeddingMapFnTest, WriteConfigurationMapMissingModelArtifactsConfigId) {
  std::string temp_dir = std::string(temp_dir_);
  EmbeddingFnFactoryProvider provider =
      CreateEmbeddingFnFactoryProvider(factory_, temp_dir);

  WriteConfigurationMap empty_map;
  EXPECT_THAT(provider(init_config_, Any(), empty_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(EmbeddingMapFnTest, InvalidArchivePath) {
  std::string temp_dir = std::string(temp_dir_);
  EmbeddingFnFactoryProvider provider =
      CreateEmbeddingFnFactoryProvider(factory_, temp_dir);

  WriteConfigurationMap config_map;
  config_map[std::string(kConfigId)] = "/tmp/DO_NOT_EXIST.zip";
  EXPECT_THAT(provider(init_config_, Any(), config_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(EmbeddingMapFnTest, InitializeReplicaInitializeRuntimeAndModel) {
  auto fn = InitializeFn();

  EXPECT_CALL(*delegate_raw_ptr_, InitializeRuntime());
  EXPECT_CALL(*delegate_raw_ptr_, InitializeModel(_)).WillOnce(Return(true));
  EXPECT_THAT(fn->InitializeReplica(Any(), context_), IsOk());
}

TEST_F(EmbeddingMapFnTest, InitializeModelFailed) {
  auto fn = InitializeFn();

  EXPECT_CALL(*delegate_raw_ptr_, InitializeRuntime());
  EXPECT_CALL(*delegate_raw_ptr_, InitializeModel(_)).WillOnce(Return(false));
  EXPECT_THAT(fn->InitializeReplica(Any(), context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(EmbeddingMapFnTest, FinalizeReplicaFinalizeRuntime) {
  auto fn = InitializeFn();
  EXPECT_CALL(*delegate_raw_ptr_, FinalizeRuntime());
  EXPECT_THAT(fn->FinalizeReplica(Any(), context_), IsOk());
}

TEST_F(EmbeddingMapFnTest, WriteSuccess) {
  auto fn = InitializeFn();

  std::string input = "winter is coming.";
  std::vector<std::vector<float>> output{{0.43, 5.46, 0.23}};
  EXPECT_CALL(*delegate_raw_ptr_,
              GenerateEmbeddings(ElementsAre(input), Eq(std::nullopt)))
      .WillOnce(Return(output));
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv), Return(true)));

  WriteRequest request;
  BlobHeader header;
  header.set_blob_id(kBlobId);
  header.set_key_id(kKeyId);
  header.set_access_policy_sha256(kReencryptionPolicyHash);
  *request.mutable_first_request_metadata() = CreateBlobMetadata(header);

  auto checkpoint = CreateInputCheckpoint({input});
  CHECK_OK(checkpoint);
  EXPECT_THAT(fn->Write(request, *checkpoint, context_), IsOk());

  auto expected_checkpoint = CreateOutputCheckpoint(output);
  CHECK_OK(expected_checkpoint);
  EXPECT_THAT(emitted_kv.data, *expected_checkpoint);
  EXPECT_THAT(emitted_kv.blob_id, kBlobId);
}

TEST_F(EmbeddingMapFnTest, MapSuccessWithPrompt) {
  std::optional<std::string> prompt = "Summarize this text:";
  auto fn = InitializeFn(prompt);
  std::string input = "winter is coming.";
  std::vector<std::vector<float>> output{{0.43, 5.46, 0.23}};
  EXPECT_CALL(*delegate_raw_ptr_,
              GenerateEmbeddings(ElementsAre(input), Eq(prompt)))
      .WillOnce(Return(output));
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv), Return(true)));

  WriteRequest request;
  BlobHeader header;
  header.set_blob_id(kBlobId);
  header.set_key_id(kKeyId);
  header.set_access_policy_sha256(kReencryptionPolicyHash);
  *request.mutable_first_request_metadata() = CreateBlobMetadata(header);

  auto checkpoint = CreateInputCheckpoint({input});
  CHECK_OK(checkpoint);
  EXPECT_THAT(fn->Write(request, *checkpoint, context_), IsOk());

  auto expected_checkpoint = CreateOutputCheckpoint(output);
  CHECK_OK(expected_checkpoint);
  EXPECT_THAT(emitted_kv.data, *expected_checkpoint);
  EXPECT_THAT(emitted_kv.blob_id, kBlobId);
}

TEST_F(EmbeddingMapFnTest, MapFailsWhenGenerateEmbeddingFails) {
  auto fn = InitializeFn();

  std::string input = "winter is coming.";
  EXPECT_CALL(*delegate_raw_ptr_,
              GenerateEmbeddings(ElementsAre(input), Eq(std::nullopt)))
      .WillOnce(Return(absl::InvalidArgumentError("Internal failure.")));

  WriteRequest request;
  BlobHeader header;
  header.set_blob_id(kBlobId);
  header.set_key_id(kKeyId);
  header.set_access_policy_sha256(kReencryptionPolicyHash);
  *request.mutable_first_request_metadata() = CreateBlobMetadata(header);

  auto checkpoint = CreateInputCheckpoint({input});
  CHECK_OK(checkpoint);
  auto result = fn->Write(request, *checkpoint, context_);
  EXPECT_EQ(result->status().code(),
            static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(EmbeddingMapFnTest, EmitFailed) {
  auto fn = InitializeFn();

  std::string input = "winter is coming.";
  std::vector<std::vector<float>> output{{0.43, 5.46, 0.23}};
  EXPECT_CALL(*delegate_raw_ptr_,
              GenerateEmbeddings(ElementsAre(input), Eq(std::nullopt)))
      .WillOnce(Return(output));
  Session::KV emitted_kv;
  EXPECT_CALL(context_, EmitEncrypted(0, _))
      .WillOnce(DoAll(SaveArg<1>(&emitted_kv), Return(false)));

  WriteRequest request;
  BlobHeader header;
  header.set_blob_id(kBlobId);
  header.set_key_id(kKeyId);
  header.set_access_policy_sha256(kReencryptionPolicyHash);
  *request.mutable_first_request_metadata() = CreateBlobMetadata(header);

  auto checkpoint = CreateInputCheckpoint({input});
  CHECK_OK(checkpoint);
  auto result = fn->Write(request, *checkpoint, context_);
  EXPECT_EQ(result->status().code(),
            static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace confidential_federated_compute::sentence_transformers