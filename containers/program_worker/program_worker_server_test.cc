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
#include "containers/program_worker/program_worker_server.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/struct.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::program_worker {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::fcp::confidentialcompute::ProgramWorker;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;
using ::tensorflow_federated::v0::Value;
using ::testing::Test;

constexpr absl::string_view kNoArgumentComputationPath =
    "containers/program_worker/testing/no_argument_comp.txtpb";
constexpr absl::string_view kNoArgumentComputationExpectedResultPath =
    "containers/program_worker/testing/no_argument_comp_expected_result.txtpb";
constexpr absl::string_view kServerDataCompPath =
    "containers/program_worker/testing/server_data_comp.txtpb";
constexpr absl::string_view kServerDataPath =
    "containers/program_worker/testing/server_data.txtpb";
constexpr absl::string_view kServerDataCompExpectedResultPath =
    "containers/program_worker/testing/server_data_comp_expected_result.txtpb";

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

absl::StatusOr<Value> LoadFileAsTffValue(absl::string_view path,
                                         bool is_computation = true) {
  // Before creating the std::ifstream, convert the absl::string_view to
  // std::string.
  std::string path_str(path);
  std::ifstream file_istream(path_str);
  if (!file_istream) {
    return absl::FailedPreconditionError("Error loading file: " + path_str);
  }
  std::stringstream file_stream;
  file_stream << file_istream.rdbuf();
  if (is_computation) {
    federated_language::Computation computation;
    if (!google::protobuf::TextFormat::ParseFromString(
            std::move(file_stream.str()), &computation)) {
      return absl::InvalidArgumentError(
          "Error parsing TFF Computation from file.");
    }
    Value value;
    *value.mutable_computation() = std::move(computation);
    return value;
  } else {
    Value value;
    if (!google::protobuf::TextFormat::ParseFromString(
            std::move(file_stream.str()), &value)) {
      return absl::InvalidArgumentError(
          "Error parsing TFF Federated from file.");
    }
    return value;
  }
}

SessionConfig* TestConfigAttestedNNClient() {
  auto verifier = bindings::new_fake_attestation_verifier(
      ffi_bindings::BytesView(kFakeEvent),
      ffi_bindings::BytesView(kFakePlatform));

  return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                              HandshakeType::kNoiseNN)
      .AddPeerVerifier(kFakeAttesterId, verifier)
      .Build();
}

SessionConfig* TestConfigAttestedNNServer() {
  auto signing_key = bindings::new_random_signing_key();
  auto verifying_bytes = bindings::signing_key_verifying_key_bytes(signing_key);

  auto fake_evidence =
      bindings::new_fake_evidence(ffi_bindings::BytesView(verifying_bytes),
                                  ffi_bindings::BytesView(kFakeEvent));
  ffi_bindings::free_rust_bytes_contents(verifying_bytes);
  auto attester =
      bindings::new_simple_attester(ffi_bindings::BytesView(fake_evidence));
  if (attester.error != nullptr) {
    LOG(FATAL) << "Failed to create attester:"
               << ffi_bindings::ErrorIntoStatus(attester.error);
  }
  ffi_bindings::free_rust_bytes_contents(fake_evidence);

  auto fake_endorsements =
      bindings::new_fake_endorsements(ffi_bindings::BytesView(kFakePlatform));
  auto endorser =
      bindings::new_simple_endorser(ffi_bindings::BytesView(fake_endorsements));
  if (endorser.error != nullptr) {
    LOG(FATAL) << "Failed to create attester:" << attester.error;
  }

  ffi_bindings::free_rust_bytes_contents(fake_endorsements);

  auto builder = SessionConfigBuilder(AttestationType::kSelfUnidirectional,
                                      HandshakeType::kNoiseNN)
                     .AddSelfAttester(kFakeAttesterId, attester.result)
                     .AddSelfEndorser(kFakeAttesterId, endorser.result)
                     .AddSessionBinder(kFakeAttesterId, signing_key);

  bindings::free_signing_key(signing_key);

  return builder.Build();
}

class ProgramWorkerTeeServerTest : public Test {
 public:
  ProgramWorkerTeeServerTest() {
    int port;
    const std::string server_address = "[::1]:";
    auto service = ProgramWorkerTee::Create(TestConfigAttestedNNServer());
    CHECK_OK(service);
    service_ = std::move(service.value());
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ProgramWorker::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~ProgramWorkerTeeServerTest() override {
    if (server_ != nullptr) {
      server_->Shutdown();
      server_->Wait();
    }
  }

  absl::StatusOr<std::unique_ptr<ClientSession>>
  CreateClientSessionAndDoHandshake() {
    FCP_ASSIGN_OR_RETURN(auto client_session,
                         ClientSession::Create(TestConfigAttestedNNClient()));

    while (!client_session->IsOpen()) {
      FCP_ASSIGN_OR_RETURN(auto init_request,
                           client_session->GetOutgoingMessage());
      if (!init_request.has_value()) {
        return absl::InternalError("init_request doesn't have value.");
      }
      ComputationRequest request;
      request.mutable_computation()->PackFrom(init_request.value());
      ComputationResponse response;
      {
        grpc::ClientContext client_context;
        auto status = stub_->Execute(&client_context, request, &response);
        if (!status.ok()) {
          return absl::InternalError("Failed to execute request on server.");
        }
      }
      if (response.has_result()) {
        SessionResponse init_response;
        if (!response.result().UnpackTo(&init_response)) {
          return absl::InternalError(
              "Failed to unpack response to init_response.");
        }
        FCP_RETURN_IF_ERROR(client_session->PutIncomingMessage(init_response));
      }
    }

    // Return an open client session.
    return client_session;
  }

 protected:
  std::unique_ptr<ProgramWorkerTee> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ProgramWorker::Stub> stub_;
};

TEST_F(ProgramWorkerTeeServerTest, ExecuteReturnsInvalidArgumentError) {
  grpc::ClientContext context;
  ComputationRequest request;
  google::protobuf::Value string_value;
  string_value.set_string_value("test");
  request.mutable_computation()->PackFrom(string_value);
  ComputationResponse response;

  auto status = stub_->Execute(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ProgramWorkerTeeServerTest, ExecuteHandshakeReturnsSessionResponse) {
  auto client_session = CreateClientSessionAndDoHandshake();
  ASSERT_TRUE(client_session.ok());
}

TEST_F(ProgramWorkerTeeServerTest, ExecuteNoArgumentComputationReturnsResult) {
  auto client_session = CreateClientSessionAndDoHandshake();
  ASSERT_TRUE(client_session.ok());

  TffSessionConfig tff_comp_request;
  auto function = LoadFileAsTffValue(kNoArgumentComputationPath);
  ASSERT_TRUE(function.ok());
  *tff_comp_request.mutable_function() = *function;
  tff_comp_request.set_num_clients(3);
  tff_comp_request.set_output_access_policy_node_id(1);
  tff_comp_request.set_max_concurrent_computation_calls(1);
  PlaintextMessage plaintext_comp_request;
  plaintext_comp_request.set_plaintext(tff_comp_request.SerializeAsString());
  ASSERT_TRUE((*client_session)->Write(plaintext_comp_request).ok());
  absl::StatusOr<std::optional<SessionRequest>> comp_session_request =
      (*client_session)->GetOutgoingMessage();
  ASSERT_TRUE(comp_session_request.ok());

  ComputationRequest comp_request;
  comp_request.mutable_computation()->PackFrom(comp_session_request->value());
  ComputationResponse comp_response;
  {
    grpc::ClientContext context;
    auto comp_status = stub_->Execute(&context, comp_request, &comp_response);
    ASSERT_TRUE(comp_status.ok());
  }

  SessionResponse comp_session_response;
  ASSERT_TRUE(comp_response.result().UnpackTo(&comp_session_response));
  ASSERT_TRUE(
      (*client_session)->PutIncomingMessage(comp_session_response).ok());
  auto decrypted_comp_response = (*client_session)->Read();
  ASSERT_TRUE(decrypted_comp_response.ok());

  Value result;
  bool parse_success =
      result.ParseFromString(decrypted_comp_response->value().plaintext());
  ASSERT_TRUE(parse_success);
  auto expected_result =
      LoadFileAsTffValue(kNoArgumentComputationExpectedResultPath, false);
  ASSERT_TRUE(expected_result.ok());
  ASSERT_EQ(result.SerializeAsString(), expected_result->SerializeAsString());
}

TEST_F(ProgramWorkerTeeServerTest, ExecuteServerDataCompReturnsResult) {
  auto client_session = CreateClientSessionAndDoHandshake();
  ASSERT_TRUE(client_session.ok());

  TffSessionConfig tff_comp_request;
  auto function = LoadFileAsTffValue(kServerDataCompPath);
  ASSERT_TRUE(function.ok());
  auto arg = LoadFileAsTffValue(kServerDataPath, false);
  ASSERT_TRUE(arg.ok());
  *tff_comp_request.mutable_initial_arg() = *arg;
  *tff_comp_request.mutable_function() = *function;
  tff_comp_request.set_num_clients(3);
  tff_comp_request.set_output_access_policy_node_id(1);
  tff_comp_request.set_max_concurrent_computation_calls(1);
  PlaintextMessage plaintext_comp_request;
  plaintext_comp_request.set_plaintext(tff_comp_request.SerializeAsString());
  ASSERT_TRUE((*client_session)->Write(plaintext_comp_request).ok());
  absl::StatusOr<std::optional<SessionRequest>> comp_session_request =
      (*client_session)->GetOutgoingMessage();
  ASSERT_TRUE(comp_session_request.ok());

  ComputationRequest comp_request;
  comp_request.mutable_computation()->PackFrom(comp_session_request->value());
  ComputationResponse comp_response;
  {
    grpc::ClientContext context;
    auto comp_status = stub_->Execute(&context, comp_request, &comp_response);
    ASSERT_TRUE(comp_status.ok());
  }

  SessionResponse comp_session_response;
  ASSERT_TRUE(comp_response.result().UnpackTo(&comp_session_response));
  ASSERT_TRUE(
      (*client_session)->PutIncomingMessage(comp_session_response).ok());
  auto decrypted_comp_response = (*client_session)->Read();
  ASSERT_TRUE(decrypted_comp_response.ok());

  Value result;
  bool parse_success =
      result.ParseFromString(decrypted_comp_response->value().plaintext());
  ASSERT_TRUE(parse_success);
  auto expected_result =
      LoadFileAsTffValue(kServerDataCompExpectedResultPath, false);
  ASSERT_TRUE(expected_result.ok());
  ASSERT_EQ(result.SerializeAsString(), expected_result->SerializeAsString());
}

}  // namespace

}  // namespace confidential_federated_compute::program_worker
