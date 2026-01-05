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

#include "containers/batched_inference/batched_inference_server.h"

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
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::crypto_test_utils::GenerateKeyPair;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
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
using ::testing::Return;
using ::testing::Test;

static const std::string kKeyId = "test_key_id";
static const std::string kPolicyHash = "hash_1";

class MockBatchedInferenceProvider : public BatchedInferenceProvider {
 public:
  MOCK_METHOD((std::vector<absl::StatusOr<std::string>>), DoBatchedInference,
              (std::vector<std::string> prompts), (override));
};

class BatchedInferenceServerTest : public ::testing::Test {
 protected:
  BatchedInferenceServerTest() {}

  void SetUp() override {
    auto encryption_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    server_public_key_ = encryption_handle->GetSerializedPublicKey();
    mock_batched_inference_provider_ =
        std::make_shared<NiceMock<MockBatchedInferenceProvider>>();
    absl::StatusOr<std::unique_ptr<BatchedInferenceServer>> server =
        CreateBatchedInferenceServer(
            mock_batched_inference_provider_, 0,
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
    mock_batched_inference_provider_.reset();
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
    associated_data.add_authorized_logical_pipeline_policies_hashes(
        kPolicyHash);

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
    header.set_access_policy_sha256(kPolicyHash);
    header.set_key_id(kKeyId);
    std::string aad = header.SerializeAsString();

    fcp::confidential_compute::MessageEncryptor encryptor;
    auto encrypt_res =
        encryptor.Encrypt(inference_data, session_pub_key_cose, aad).value();

    SessionRequest write_req;
    auto* write = write_req.mutable_write();
    write->set_data(encrypt_res.ciphertext);
    write->set_commit(true);

    auto* metadata = write->mutable_first_request_metadata();
    metadata->set_compression_type(
        fcp::confidentialcompute::BlobMetadata::COMPRESSION_TYPE_NONE);
    metadata->set_total_size_bytes(encrypt_res.ciphertext.size());

    auto* hpke = metadata->mutable_hpke_plus_aead_data();
    hpke->set_ciphertext_associated_data(aad);
    hpke->set_encrypted_symmetric_key(encrypt_res.encrypted_symmetric_key);
    hpke->set_encapsulated_public_key(encrypt_res.encapped_key);
    hpke->mutable_kms_symmetric_key_associated_data()->set_record_header(aad);

    ASSERT_TRUE(session_stream->Write(write_req));
  }

  // Push one commit message to a session initilaized with InitializeSession.
  void WriteCommitToSession(SessionStream& session_stream) {
    SessionRequest commit_req;
    commit_req.mutable_commit();
    ASSERT_TRUE(session_stream->Write(commit_req));
  }

  // Read all replies until the first non-read, and accumulate the reads on the
  // output vector.
  absl::StatusOr<std::unique_ptr<SessionResponse>> ReadResponsesFromSession(
      SessionStream& session_stream,
      fcp::confidential_compute::MessageDecryptor& decryptor,
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
        const std::string& ciphertext = session_response->read().data();
        auto decrypted_result = decryptor.Decrypt(
            ciphertext,
            read_metadata.hpke_plus_aead_data().ciphertext_associated_data(),
            read_metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
            read_metadata.hpke_plus_aead_data()
                .kms_symmetric_key_associated_data()
                .record_header(),
            read_metadata.hpke_plus_aead_data().encapsulated_public_key(),
            kKeyId);
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
  std::shared_ptr<NiceMock<MockBatchedInferenceProvider>>
      mock_batched_inference_provider_;
  std::unique_ptr<BatchedInferenceServer> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
};

TEST_F(BatchedInferenceServerTest, NoWrites) {
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, SingleWriteNoCommit) {
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  fcp::confidential_compute::MessageDecryptor decryptor(
      std::vector<absl::string_view>(
          {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  WriteInferenceDataToSession(session_stream, session_pub_key_cose, "test_blob",
                              "some_inference_data");
  std::vector<std::string> responses;
  absl::StatusOr<std::unique_ptr<SessionResponse>> read_result =
      ReadResponsesFromSession(session_stream, decryptor, &responses);
  EXPECT_THAT(read_result.status(), IsOk());
  EXPECT_TRUE(read_result.value()->has_write());
  EXPECT_TRUE(responses.empty());
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleWritesNoCommit) {
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(0);
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  fcp::confidential_compute::MessageDecryptor decryptor(
      std::vector<absl::string_view>(
          {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  const int kNumWrites = 10;
  for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
    WriteInferenceDataToSession(session_stream, session_pub_key_cose,
                                "test_blob", "some_inference_data");
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
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(1)
      .WillOnce(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  fcp::confidential_compute::MessageDecryptor decryptor(
      std::vector<absl::string_view>(
          {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  WriteInferenceDataToSession(session_stream, session_pub_key_cose, "test_blob",
                              "some_inference_data");
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
  EXPECT_EQ(responses[0], "Processed: some_inference_data");
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleWritesWithCommit) {
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(1)
      .WillOnce(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  fcp::confidential_compute::MessageDecryptor decryptor(
      std::vector<absl::string_view>(
          {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  std::vector<std::string> responses;
  const int kNumWrites = 10;
  for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
    WriteInferenceDataToSession(session_stream, session_pub_key_cose,
                                "test_blob",
                                absl::StrCat("some_inference_data_", write_no));
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
              absl::StrCat("Processed: some_inference_data_", write_no));
  }
  FinalizeSession(session_stream);
}

TEST_F(BatchedInferenceServerTest, MultipleBatches) {
  const int kNumBatches = 5;
  EXPECT_CALL(*mock_batched_inference_provider_, DoBatchedInference(_))
      .Times(kNumBatches)
      .WillRepeatedly(Invoke([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& prompt : prompts) {
          results.push_back("Processed: " + prompt);
        }
        return results;
      }));
  auto [session_pub_key_cose, session_priv_key_cose] = GenerateKeyPair(kKeyId);
  fcp::confidential_compute::MessageDecryptor decryptor(
      std::vector<absl::string_view>(
          {session_priv_key_cose, session_priv_key_cose}));
  grpc::ClientContext session_context;
  SessionStream session_stream;
  InitializeSession(session_pub_key_cose, session_priv_key_cose,
                    &session_context, &session_stream);
  for (int batch_no = 1; batch_no <= kNumBatches; ++batch_no) {
    std::vector<std::string> responses;
    const int kNumWrites = 5;
    for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
      WriteInferenceDataToSession(
          session_stream, session_pub_key_cose, "test_blob",
          absl::StrCat("some_inference_data_", batch_no, "_", write_no));
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
                absl::StrCat("Processed: some_inference_data_", batch_no, "_",
                             write_no));
    }
  }
  FinalizeSession(session_stream);
}

}  // namespace
}  // namespace confidential_federated_compute::batched_inference
