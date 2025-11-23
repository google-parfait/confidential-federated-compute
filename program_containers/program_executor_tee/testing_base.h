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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/escaping.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "program_executor_tee/confidential_transform_server.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/python_manager.h"
#include "proto/attestation/reference_value.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigurationMetadata;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ProgramExecutorTeeConfigConstraints;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::attestation::v1::ReferenceValues;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::testing::NiceMock;
using ::testing::Test;

inline constexpr int kMaxNumSessions = 8;

class PythonEnvironment : public ::testing::Environment {
 public:
  // Called once before the very first test case starts.
  void SetUp() override {
    // This starts the dedicated python execution thread.
    PythonManager::GetInstance().Start();
  }

  // Called once after the very last test case finishes.
  void TearDown() override {
    // This stops the dedicated python execution thread.
    PythonManager::GetInstance().Stop();
  }
};

void ReadTextProtoOrDie(const std::string& path,
                        google::protobuf::Message* message) {
  std::filesystem::path my_path = std::filesystem::current_path() / path;

  std::ifstream file(my_path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open proto file: " << my_path;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(buffer.str(), message)) {
    LOG(FATAL) << "Failed to parse textproto file: " << my_path;
  }
}

ProgramExecutorTeeInitializeConfig CreateProgramExecutorTeeInitializeConfig(
    std::string program, std::vector<std::string> client_ids = {},
    std::string client_data_dir = "", std::string outgoing_server_address = "",
    std::optional<std::string> worker_reference_values_path = std::nullopt) {
  ProgramExecutorTeeInitializeConfig init_config;
  init_config.set_program(absl::Base64Escape(program));
  init_config.set_outgoing_server_address(outgoing_server_address);
  init_config.set_attester_id("fake_attester");
  init_config.set_client_data_dir(client_data_dir);
  if (worker_reference_values_path) {
    ReferenceValues worker_reference_values;
    ReadTextProtoOrDie((*worker_reference_values_path),
                       &worker_reference_values);
    *init_config.mutable_reference_values() = worker_reference_values;
  }
  for (const std::string& client_id : client_ids) {
    init_config.add_client_ids(client_id);
  }
  return init_config;
}

ProgramExecutorTeeConfigConstraints CreateProgramExecutorTeeConfigConstraints(
    std::string program,
    std::optional<std::string> worker_reference_values_path = std::nullopt) {
  ProgramExecutorTeeConfigConstraints config_constraints;
  config_constraints.set_program(absl::Base64Escape(program));
  if (worker_reference_values_path) {
    ReferenceValues worker_reference_values;
    ReadTextProtoOrDie((*worker_reference_values_path),
                       &worker_reference_values);
    config_constraints.set_worker_reference_values(
        absl::Base64Escape(worker_reference_values.SerializeAsString()));
  }
  config_constraints.set_num_runs(5);
  return config_constraints;
}

template <typename T>
class ProgramExecutorTeeTest : public Test {
 public:
  ProgramExecutorTeeTest() {
    const std::string localhost = "[::1]:";

    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<T>(
        std::make_unique<NiceMock<crypto_test_utils::MockSigningKeyHandle>>(),
        std::move(encryption_key_handle));

    int confidential_transform_server_port;
    ServerBuilder builder;
    builder.AddListeningPort(localhost + "0", grpc::InsecureServerCredentials(),
                             &confidential_transform_server_port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "ConfidentialTransform server listening on "
              << localhost + std::to_string(confidential_transform_server_port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(grpc::CreateChannel(
        localhost + std::to_string(confidential_transform_server_port),
        grpc::InsecureChannelCredentials()));

    int data_read_write_service_port;
    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(localhost + "0",
                                             grpc::InsecureServerCredentials(),
                                             &data_read_write_service_port);
    data_read_write_builder.RegisterService(&fake_data_read_write_service_);
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    data_read_write_server_address_ =
        localhost + std::to_string(data_read_write_service_port);
    LOG(INFO) << "DataReadWrite server listening on "
              << data_read_write_server_address_ << std::endl;
  }

  ~ProgramExecutorTeeTest() override { server_->Shutdown(); }

 protected:
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
  std::unique_ptr<T> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;

  std::string data_read_write_server_address_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
};

template <typename T>
class ProgramExecutorTeeSessionTest
    : public ProgramExecutorTeeTest<T>,
      public ::testing::WithParamInterface<bool> {
 public:
  static std::string TestNameSuffix(
      const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "WithKms" : "NoKms";
  }

  void CreateSession(
      std::string program, bool use_kms,
      std::vector<std::string> client_ids = {},
      std::string client_data_dir = "",
      std::map<std::string, std::string> file_id_to_filepath = {}) {
    grpc::ClientContext configure_context;

    std::vector<StreamInitializeRequest> requests;
    for (const auto& [file_id, filepath] : file_id_to_filepath) {
      StreamInitializeRequest request;
      std::ifstream file(filepath);
      CHECK(file.is_open());
      auto file_size = std::filesystem::file_size(filepath);
      std::string file_content(file_size, '\0');
      file.read(file_content.data(), file_size);
      request.mutable_write_configuration()->set_data(file_content);
      ConfigurationMetadata* metadata = request.mutable_write_configuration()
                                            ->mutable_first_request_metadata();
      metadata->set_configuration_id(file_id);
      metadata->set_total_size_bytes(file_size);
      request.mutable_write_configuration()->set_commit(true);
      file.close();
      requests.push_back(std::move(request));
    }

    ProgramExecutorTeeInitializeConfig config =
        CreateProgramExecutorTeeInitializeConfig(
            program, client_ids, client_data_dir,
            this->data_read_write_server_address_);
    StreamInitializeRequest stream_initialize_request;
    InitializeRequest* initialize_request =
        stream_initialize_request.mutable_initialize_request();
    initialize_request->set_max_num_sessions(kMaxNumSessions);
    initialize_request->mutable_configuration()->PackFrom(config);
    if (use_kms) {
      AuthorizeConfidentialTransformResponse::ProtectedResponse
          protected_response;
      *protected_response.add_result_encryption_keys() =
          "result_encryption_key";
      AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
      associated_data.mutable_config_constraints()->PackFrom(
          CreateProgramExecutorTeeConfigConstraints(program));
      associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
      auto encrypted_request =
          this->oak_client_encryptor_
              ->Encrypt(protected_response.SerializeAsString(),
                        associated_data.SerializeAsString())
              .value();
      *initialize_request->mutable_protected_response() =
          std::move(encrypted_request);
    }
    requests.push_back(std::move(stream_initialize_request));

    InitializeResponse response;
    std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
        this->stub_->StreamInitialize(&configure_context, &response);
    for (const StreamInitializeRequest& request : requests) {
      writer->Write(request);
    }
    CHECK(writer->WritesDone());
    CHECK(writer->Finish().ok());

    public_key_ = response.public_key();

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

    stream_ = this->stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
    nonce_generator_ =
        std::make_unique<NonceGenerator>(session_response.configure().nonce());
  }

 protected:
  bool UseKms() const { return GetParam(); }

  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<NonceGenerator> nonce_generator_;
  std::string public_key_;
};

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
