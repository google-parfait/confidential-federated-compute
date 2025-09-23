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
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "gtest/gtest.h"
#include "program_executor_tee/confidential_transform_server.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigurationMetadata;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::testing::Test;

// This file of base test classes exists because there can only be one test per
// file that exercises the pybind11::scoped_interpreter code in
// FinalizeSession. This is because all test cases in a single file run in the
// same process and pybind11::scoped_interpreter is only allowed to be used
// once per process (third-party extension modules like numpy do not load
// correctly if pybind11::scoped_interpreter is used a second time). Any test
// that exercises pybind11::scoped_interpreter should extend
// ProgramExecutorTeeSessionTest but be added to a new integration test file.

inline constexpr int kMaxNumSessions = 8;

template <typename T>
class ProgramExecutorTeeTest : public Test {
 public:
  ProgramExecutorTeeTest() {
    const std::string localhost = "[::1]:";

    int confidential_transform_server_port;
    ServerBuilder builder;
    builder.AddListeningPort(localhost + "0", grpc::InsecureServerCredentials(),
                             &confidential_transform_server_port);
    builder.RegisterService(&service_);
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
  T service_{std::make_unique<
      testing::NiceMock<crypto_test_utils::MockSigningKeyHandle>>()};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;

  std::string data_read_write_server_address_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
};

template <typename T>
class ProgramExecutorTeeSessionTest : public ProgramExecutorTeeTest<T> {
 public:
  void CreateSession(
      std::string program, std::vector<std::string> client_ids = {},
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

    ProgramExecutorTeeInitializeConfig config;
    config.set_program(absl::Base64Escape(program));
    config.set_outgoing_server_address(this->data_read_write_server_address_);
    config.set_attester_id("fake_attester");
    config.set_client_data_dir(client_data_dir);
    for (const std::string& client_id : client_ids) {
      config.add_client_ids(client_id);
    }
    StreamInitializeRequest stream_initialize_request;
    InitializeRequest* initialize_request =
        stream_initialize_request.mutable_initialize_request();
    initialize_request->set_max_num_sessions(kMaxNumSessions);
    initialize_request->mutable_configuration()->PackFrom(config);
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
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<NonceGenerator> nonce_generator_;
  std::string public_key_;
};

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
