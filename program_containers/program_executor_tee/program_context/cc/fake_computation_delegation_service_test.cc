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

#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"

#include <memory>
#include <string>

#include "cc/ffi/bytes_view.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "program_executor_tee/proto/executor_wrapper.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidentialcompute::outgoing::ComputationDelegation;
using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
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
using ::testing::Test;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";
constexpr int kNumClients = 5;
const std::vector<std::string> kWorkerBns = {"worker_1", "worker_2"};

SessionConfig* TestConfigAttestedNNClient() {
  auto verifier = bindings::new_fake_attestation_verifier(
      ffi_bindings::BytesView(kFakeEvent),
      ffi_bindings::BytesView(kFakePlatform));

  return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                              HandshakeType::kNoiseNN)
      .AddPeerVerifier(kFakeAttesterId, verifier)
      .Build();
}

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return federated_language_jax::CreateXLAExecutor();
}

class FakeComputationDelegationServiceTest : public Test {
 public:
  FakeComputationDelegationServiceTest()
      : fake_computation_delegation_service_(kWorkerBns, CreateExecutor) {
    const std::string server_address = "[::1]:";

    ServerBuilder computation_delegation_builder;
    computation_delegation_builder.AddListeningPort(
        absl::StrCat(server_address, 0), grpc::InsecureServerCredentials(),
        &port_);
    computation_delegation_builder.RegisterService(
        &fake_computation_delegation_service_);
    fake_computation_delegation_server_ =
        computation_delegation_builder.BuildAndStart();
    LOG(INFO) << "ComputationDelegation server listening on "
              << server_address + std::to_string(port_) << std::endl;

    stub_ = ComputationDelegation::NewStub(
        grpc::CreateChannel(absl::StrCat(server_address, port_),
                            grpc::InsecureChannelCredentials()));
  }

  ~FakeComputationDelegationServiceTest() override {
    fake_computation_delegation_server_->Shutdown();
  }

  absl::StatusOr<std::unique_ptr<ClientSession>>
  CreateClientSessionAndDoHandshake(std::string worker_bns) {
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
      request.set_worker_bns(worker_bns);
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
  int port_;
  FakeComputationDelegationService fake_computation_delegation_service_;
  std::unique_ptr<Server> fake_computation_delegation_server_;
  std::unique_ptr<ComputationDelegation::Stub> stub_;
};

TEST_F(FakeComputationDelegationServiceTest, OpenSessionForEachWorker) {
  for (const auto& worker_bns : kWorkerBns) {
    auto client_session = CreateClientSessionAndDoHandshake(worker_bns);
    ASSERT_TRUE(client_session.ok());
  }
}

TEST_F(FakeComputationDelegationServiceTest, ExecuteSuccess) {
  std::string worker_bns = kWorkerBns[0];
  auto client_session = CreateClientSessionAndDoHandshake(worker_bns);
  ASSERT_TRUE(client_session.ok());

  // Create a ExecutorGroupRequest for a GetExecutorRequest.
  executor_wrapper::ExecutorGroupRequest request;
  tensorflow_federated::v0::GetExecutorRequest* get_executor_request =
      request.mutable_get_executor_request();
  auto cardinalities = get_executor_request->mutable_cardinalities()->Add();
  cardinalities->mutable_placement()->set_uri("clients");
  cardinalities->set_cardinality(kNumClients);

  // Interact with the client session to get the SessionRequest that wraps the
  // ExecutorGroupRequest.
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(request.SerializeAsString());
  ASSERT_TRUE((*client_session)->Write(plaintext_request).ok());
  absl::StatusOr<std::optional<SessionRequest>> comp_session_request =
      (*client_session)->GetOutgoingMessage();
  ASSERT_TRUE(comp_session_request.ok());

  // Wrap the SessionRequest in a ComputationRequest and send it to the stub.
  ComputationRequest comp_request;
  comp_request.mutable_computation()->PackFrom(comp_session_request->value());
  comp_request.set_worker_bns(worker_bns);
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

  // Retrieve a ExecutorGroupResponse.
  executor_wrapper::ExecutorGroupResponse response;
  bool parse_success =
      response.ParseFromString(decrypted_comp_response->value().plaintext());
  ASSERT_TRUE(parse_success);
  ASSERT_TRUE(response.has_get_executor_response());
}

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
