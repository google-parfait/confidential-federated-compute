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

#include "containers/common/inference/batched_inference_server.h"

#include <execinfo.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/blob_metadata.h"
#include "containers/common/inference/batched_inference_engine.h"
#include "containers/common/inference/batched_inference_test_utils.h"
#include "containers/common/io/tabular/input.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/compression.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/private_logger_uploads_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::Decryptor;
using ::confidential_federated_compute::GetKeyIdFromMetadata;
using ::confidential_federated_compute::crypto_test_utils::GenerateKeyPair;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::google::protobuf::Any;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::testing::_;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Invoke;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Property;
using ::testing::Return;
using ::testing::Test;

static const std::string kKeyId = "test_key_id";

class MockBatchedInferenceEngine : public BatchedInferenceEngine {
 public:
  MOCK_METHOD((std::vector<absl::StatusOr<std::string>>), DoBatchedInference,
              (std::vector<std::string> prompts), (override));
};

class MockBatchedInferenceEngineProvider
    : public BatchedInferenceEngineProvider {
 public:
  MOCK_METHOD((std::shared_ptr<BatchedInferenceEngine>),
              GetEngineForInferenceConfig,
              (const fcp::confidentialcompute::InferenceConfiguration&
                   inference_config),
              (override));
};

class BatchedInferenceServerTest : public ::testing::Test {
 protected:
  BatchedInferenceServerTest() {}

  void SetUp() override {
    auto encryption_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    server_public_key_ = encryption_handle->GetSerializedPublicKey();
    mock_batched_inference_engine_ =
        std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
    mock_batched_inference_engine_provider_ =
        std::make_shared<NiceMock<MockBatchedInferenceEngineProvider>>();
    EXPECT_CALL(*mock_batched_inference_engine_provider_,
                GetEngineForInferenceConfig(_))
        .WillRepeatedly(Return(mock_batched_inference_engine_));
    absl::StatusOr<std::unique_ptr<BatchedInferenceServer>> server =
        CreateBatchedInferenceServer(
            mock_batched_inference_engine_provider_, 0,
            std::make_unique<confidential_federated_compute::crypto_test_utils::
                                 MockSigningKeyHandle>(),
            std::move(encryption_handle));
    CHECK_OK(server);
    server_ = std::move(server.value());
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel("[::1]:" + std::to_string(server_->port()),
                            grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    stub_.reset();
    server_.reset();
    mock_batched_inference_engine_provider_.reset();
    mock_batched_inference_engine_.reset();
  }

  typedef std::unique_ptr<
      grpc::ClientReaderWriter<fcp::confidentialcompute::SessionRequest,
                               fcp::confidentialcompute::SessionResponse>>
      SessionStream;

  // Initialize the session for use in a test.
  void InitializeSession(const std::string& session_pub_key_cose,
                         const std::string& session_priv_key_cose,
                         grpc::ClientContext* session_context,
                         SessionStream* ptr_session_stream) {
    auto handshake_encryptor =
        ClientEncryptor::Create(server_public_key_).value();

    AuthorizeConfidentialTransformResponse::ProtectedResponse protected_resp;
    protected_resp.add_result_encryption_keys(session_pub_key_cose);
    protected_resp.add_decryption_keys(session_priv_key_cose);

    AuthorizeConfidentialTransformResponse::AssociatedData associated_data;

    auto encrypted_handshake =
        handshake_encryptor
            ->Encrypt(protected_resp.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();

    grpc::ClientContext init_context;
    InitializeResponse init_response;
    auto init_stream = stub_->StreamInitialize(&init_context, &init_response);

    StreamInitializeRequest init_request;
    init_request.mutable_initialize_request()->set_max_num_sessions(1);
    *init_request.mutable_initialize_request()->mutable_protected_response() =
        encrypted_handshake;

    testing::AddInitConfigForTest(&init_request);

    ASSERT_TRUE(init_stream->Write(init_request));
    ASSERT_TRUE(init_stream->WritesDone());
    ASSERT_THAT(FromGrpcStatus(init_stream->Finish()), IsOk());

    auto session_stream = stub_->Session(session_context);

    SessionRequest config_req;
    config_req.mutable_configure()->set_chunk_size(1024 * 1024);
    ASSERT_TRUE(session_stream->Write(config_req));
    SessionResponse config_resp;
    ASSERT_TRUE(session_stream->Read(&config_resp));
    *ptr_session_stream = std::move(session_stream);
  }

  // Push one write message to a session initilaized with InitializeSession.
  void WriteInferenceDataToSession(SessionStream& session_stream,
                                   const std::string& session_pub_key_cose,
                                   const std::string& blob_id,
                                   const std::string& inference_data) {
    BlobHeader header;
    header.set_blob_id(blob_id);
    header.set_key_id(kKeyId);
    std::string aad = header.SerializeAsString();

    absl::StatusOr<std::string> compressed_data =
        fcp::CompressWithGzip(inference_data);
    CHECK(compressed_data.ok());

    fcp::confidential_compute::MessageEncryptor encryptor;
    auto encrypt_res =
        encryptor.Encrypt(*compressed_data, session_pub_key_cose, aad).value();

    SessionRequest write_req;
    auto* write = write_req.mutable_write();
    write->set_data(encrypt_res.ciphertext);
    write->set_commit(true);

    auto* metadata = write->mutable_first_request_metadata();
    metadata->set_compression_type(
        fcp::confidentialcompute::BlobMetadata::COMPRESSION_TYPE_GZIP);
    metadata->set_total_size_bytes(encrypt_res.ciphertext.size());

    auto* hpke = metadata->mutable_hpke_plus_aead_data();
    hpke->set_ciphertext_associated_data(aad);
    hpke->set_encrypted_symmetric_key(encrypt_res.encrypted_symmetric_key);
    hpke->set_encapsulated_public_key(encrypt_res.encapped_key);
    hpke->set_key_id(kKeyId);
    hpke->mutable_kms_symmetric_key_associated_data()
        ->mutable_associated_metadata()
        ->set_type_url(
            "type.googleapis.com/fcp.confidentialcompute.BlobHeader");
    hpke->mutable_kms_symmetric_key_associated_data()
        ->mutable_associated_metadata()
        ->set_value(aad);

    ASSERT_TRUE(session_stream->Write(write_req));
  }

  // Push one commit message to a session initialized with InitializeSession.
  void WriteCommitToSession(SessionStream& session_stream) {
    SessionRequest commit_req;
    commit_req.mutable_commit();
    ASSERT_TRUE(session_stream->Write(commit_req));
  }

  // Read all replies until the first non-read, and accumulate the reads on the
  // output vector.
  absl::StatusOr<std::unique_ptr<SessionResponse>> ReadResponsesFromSession(
      SessionStream& session_stream, Decryptor& decryptor,
      std::vector<std::string>* ptr_response_vec) {
    bool received_read_response = false;
    BlobMetadata read_metadata;
    std::unique_ptr<SessionResponse> session_response =
        std::make_unique<SessionResponse>();
    while (session_stream->Read(session_response.get())) {
      if (session_response->has_read()) {
        // Make sure we do get metadata on the first read.
        if (!received_read_response) {
          if (!session_response->read().has_first_response_metadata()) {
            return absl::InternalError("Missing first response metadata");
          }
          received_read_response = true;
        }
        // Update metadata any time there is a newer version.
        if (session_response->read().has_first_response_metadata()) {
          read_metadata = session_response->read().first_response_metadata();
        }
        ABSL_ASSIGN_OR_RETURN(std::string key_id,
                              GetKeyIdFromMetadata(read_metadata));
        auto decrypted_result = decryptor.DecryptBlob(
            read_metadata, std::string(session_response->read().data()),
            key_id);
        if (!decrypted_result.ok()) {
          return absl::InternalError(absl::StrCat(
              "Decryption error: ", decrypted_result.status().ToString()));
        }
        ptr_response_vec->push_back(*decrypted_result);
      } else {
        return std::move(session_response);
      }
    }
    return absl::InternalError("Unrechable code.");
  }

  // Final cleanup.
  void FinalizeSession(SessionStream& session_stream) {
    SessionRequest finalize_req;
    finalize_req.mutable_finalize();
    ASSERT_TRUE(session_stream->Write(finalize_req));

    SessionResponse finalize_resp;
    ASSERT_TRUE(session_stream->Read(&finalize_resp));
    ASSERT_TRUE(finalize_resp.has_finalize());

    session_stream->WritesDone();
    ASSERT_THAT(FromGrpcStatus(session_stream->Finish()), IsOk());
  }

  std::string server_public_key_;
  std::shared_ptr<NiceMock<MockBatchedInferenceEngine>>
      mock_batched_inference_engine_;
  std::shared_ptr<NiceMock<MockBatchedInferenceEngineProvider>>
      mock_batched_inference_engine_provider_;
  std::unique_ptr<BatchedInferenceServer> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
};

TEST_F(BatchedInferenceServerTest, FactoryHandlesInferenceConfig) {
  Mock::VerifyAndClearExpectations(
      mock_batched_inference_engine_provider_.get());
  InferenceConfiguration inference_config =
      testing::GetInferenceConfigForTest();
  EXPECT_CALL(
      *mock_batched_inference_engine_provider_,
      GetEngineForInferenceConfig(Property(
          &InferenceConfiguration::runtime_config,
          Property(&fcp::confidentialcompute::RuntimeConfig::max_batch_size,
                   Eq(inference_config.runtime_config().max_batch_size())))))
      .Times(1)
      .WillOnce(Invoke([this](const InferenceConfiguration& config) {
        return mock_batched_inference_engine_;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, NoWrites) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_)).Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, SingleWriteNoCommit) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_)).Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  WriteInferenceDataToSession(
      session_stream, session_pub_key_cose, "test_blob",
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"}));
  std::vector<std::string> responses;
  absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
      ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_write());
  EXPECT_TRUE(responses.empty());
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleWritesNoCommit) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_)).Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  const int kNumWrites = 10;
  for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
    WriteInferenceDataToSession(
        session_stream, session_pub_key_cose, "test_blob",
        testing::GetPrivateInferenceInputCheckpointForTest(
            {absl::StrCat("bark_", write_no)}));
    std::vector<std::string> responses;
    absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
        ReadResponsesFromSession(session_stream, decryptor, &responses);
    EXPECT_THAT(read_result.status(), IsOk());
    EXPECT_TRUE(read_result.value()->has_write());
    EXPECT_TRUE(responses.empty());
  }
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, SingleWriteWithCommit) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_))
      .Times(1)
      .WillOnce(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  WriteInferenceDataToSession(
      session_stream, session_pub_key_cose, "test_blob",
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"}));
  std::vector<std::string> responses;
  absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
      ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_write());
  EXPECT_TRUE(responses.empty());
  WriteCommitToSession(session_stream);
  read_result = ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_commit());
  EXPECT_FALSE(responses.empty());
  EXPECT_EQ(responses.size(), 1);
  EXPECT_EQ(responses[0], testing::GetPrivateInferenceOutputCheckpointForTest(
                              {"bark"}, {"Processed: Hello, bark"}));
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleWritesWithCommit) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_))
      .WillRepeatedly(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  std::vector<std::string> responses;
  const int kNumWrites = 10;
  for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
    WriteInferenceDataToSession(
        session_stream, session_pub_key_cose, "test_blob",
        testing::GetPrivateInferenceInputCheckpointForTest(
            {absl::StrCat("bark_", write_no)}));
    absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
        ReadResponsesFromSession(session_stream, decryptor, &responses);
    EXPECT_THAT(read_result.status(), IsOk());
    EXPECT_TRUE(read_result.value()->has_write());
    EXPECT_TRUE(responses.empty());
  }
  WriteCommitToSession(session_stream);
  absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
      ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_commit());
  EXPECT_FALSE(responses.empty());
  EXPECT_EQ(responses.size(), kNumWrites);
  for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
    EXPECT_EQ(responses[write_no - 1],
              testing::GetPrivateInferenceOutputCheckpointForTest(
                  {absl::StrCat("bark_", write_no)},
                  {absl::StrCat("Processed: Hello, bark_", write_no)}));
  }
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleCommits) {
  const int kNumCommits = 5;
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_))
      .WillRepeatedly(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  for (int commit_no = 1; commit_no <= kNumCommits; ++commit_no) {
    std::vector<std::string> responses;
    const int kNumWrites = 5;
    for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
      WriteInferenceDataToSession(
          session_stream, session_pub_key_cose, "test_blob",
          testing::GetPrivateInferenceInputCheckpointForTest(
              {absl::StrCat("bark_", commit_no, "_", write_no)}));
      absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
          ReadResponsesFromSession(session_stream, decryptor, &responses);
      EXPECT_THAT(read_result.status(), IsOk());
      EXPECT_TRUE(read_result.value()->has_write());
      EXPECT_TRUE(responses.empty());
    }
    WriteCommitToSession(session_stream);
    absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
        ReadResponsesFromSession(session_stream, decryptor, &responses);
    EXPECT_THAT(read_result.status(), IsOk());
    EXPECT_TRUE(read_result.value()->has_commit());
    EXPECT_FALSE(responses.empty());
    EXPECT_EQ(responses.size(), kNumWrites);
    for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
      EXPECT_EQ(responses[write_no - 1],
                testing::GetPrivateInferenceOutputCheckpointForTest(
                    {absl::StrCat("bark_", commit_no, "_", write_no)},
                    {absl::StrCat("Processed: Hello, bark_", commit_no, "_",
                                  write_no)}));
    }
  }
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, SingleWriteWithCommitPrivateLogger) {
  EXPECT_CALL(*mock_batched_inference_engine_, DoBatchedInference(_))
      .Times(1)
      .WillOnce(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));

  google::protobuf::FileDescriptorSet descriptor_set;
  auto* file_proto = descriptor_set.add_file();
  file_proto->set_name("test_log.proto");
  file_proto->set_package("test");
  file_proto->set_syntax("proto3");
  auto* msg_proto = file_proto->add_message_type();
  msg_proto->set_name("TestLogEntry");
  auto* field_proto = msg_proto->add_field();
  field_proto->set_name("transcript");
  field_proto->set_number(1);
  field_proto->set_label(
      google::protobuf::FieldDescriptorProto::LABEL_OPTIONAL);
  field_proto->set_type(google::protobuf::FieldDescriptorProto::TYPE_STRING);

  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  Decryptor decryptor(std::vector<absl::string_view>(
      {session_priv_key_cose, session_priv_key_cose}));

  auto handshake_encryptor =
      ClientEncryptor::Create(server_public_key_).value();
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_resp;
  protected_resp.add_result_encryption_keys(session_pub_key_cose);
  protected_resp.add_decryption_keys(session_priv_key_cose);
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  auto encrypted_handshake = handshake_encryptor
                                 ->Encrypt(protected_resp.SerializeAsString(),
                                           associated_data.SerializeAsString())
                                 .value();

  grpc::ClientContext init_context;
  InitializeResponse init_response;
  auto init_stream = stub_->StreamInitialize(&init_context, &init_response);
  StreamInitializeRequest init_request;
  init_request.mutable_initialize_request()->set_max_num_sessions(1);
  *init_request.mutable_initialize_request()->mutable_protected_response() =
      encrypted_handshake;

  fcp::confidentialcompute::BatchedInferenceContainerInitializeConfiguration
      container_init_config;
  *container_init_config.mutable_inference_init_config()
       ->mutable_inference_config() = testing::GetInferenceConfigForTest();
  auto* pl_config =
      container_init_config.mutable_private_logger_uploads_config();
  pl_config->set_on_device_query_name("my_query");
  pl_config->mutable_message_description()->set_message_name(
      "test.TestLogEntry");
  pl_config->mutable_message_description()->set_message_descriptor_set(
      descriptor_set.SerializeAsString());
  init_request.mutable_initialize_request()->mutable_configuration()->PackFrom(
      container_init_config);

  ASSERT_TRUE(init_stream->Write(init_request));
  ASSERT_TRUE(init_stream->WritesDone());
  ASSERT_THAT(FromGrpcStatus(init_stream->Finish()), IsOk());

  grpc::ClientContext session_context;
  SessionStream session_stream = stub_->Session(&session_context);
  SessionRequest config_req;
  config_req.mutable_configure()->set_chunk_size(1024 * 1024);
  ASSERT_TRUE(session_stream->Write(config_req));
  SessionResponse config_resp;
  ASSERT_TRUE(session_stream->Read(&config_resp));

  // Build PrivateLogger checkpoint payload.
  auto message_factory_or = FileDescriptorSetMessageFactory::Create(
      descriptor_set, "test.TestLogEntry");
  ASSERT_THAT(message_factory_or.status(), IsOk());
  auto msg1 = (*message_factory_or)->NewMessage();
  msg1->GetReflection()->SetString(
      msg1.get(), msg1->GetDescriptor()->FindFieldByName("transcript"), "bark");
  tensorflow_federated::aggregation::FederatedComputeCheckpointBuilderFactory
      builder_factory;
  auto builder = builder_factory.Create();
  const std::string entry_col_name = absl::StrCat(
      "my_query/", fcp::confidential_compute::kPrivateLoggerEntryKey);
  const std::string time_col_name = absl::StrCat(
      "my_query/", fcp::confidential_compute::kEventTimeColumnName);
  ASSERT_THAT(
      builder->Add(entry_col_name,
                   tensorflow_federated::aggregation::Tensor(
                       std::vector<std::string>{msg1->SerializeAsString()},
                       entry_col_name)),
      IsOk());
  ASSERT_THAT(builder->Add(time_col_name,
                           tensorflow_federated::aggregation::Tensor(
                               std::vector<std::string>{"2026-01-01T00:00:00Z"},
                               time_col_name)),
              IsOk());
  auto input_ckpt = builder->Build();
  ASSERT_THAT(input_ckpt.status(), IsOk());

  WriteInferenceDataToSession(session_stream, session_pub_key_cose, "test_blob",
                              std::string(input_ckpt->Flatten()));
  std::vector<std::string> responses;
  absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
      ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_write());
  EXPECT_TRUE(responses.empty());

  WriteCommitToSession(session_stream);
  read_result = ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_commit());
  ASSERT_EQ(responses.size(), 1);

  tensorflow_federated::aggregation::FederatedComputeCheckpointParserFactory
      parser_factory;
  auto parser = parser_factory.Create(absl::Cord(responses[0]));
  ASSERT_THAT(parser.status(), IsOk());
  auto out_tensors = (*parser)->LoadAllTensors();
  ASSERT_THAT(out_tensors.status(), IsOk());
  ASSERT_TRUE(out_tensors->contains("transcript"));
  EXPECT_THAT(out_tensors->at("transcript").AsSpan<absl::string_view>(),
              ::testing::ElementsAre("bark"));
  ASSERT_TRUE(out_tensors->contains("topic"));
  EXPECT_THAT(out_tensors->at("topic").AsSpan<absl::string_view>(),
              ::testing::ElementsAre("Processed: Hello, bark"));
  ASSERT_TRUE(
      out_tensors->contains(fcp::confidential_compute::kEventTimeColumnName));

  FinalizeSession(session_stream);
}

}  // namespace
}  // namespace confidential_federated_compute::inference
