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

#include "program_executor_tee/program_context/cc/noise_executor_stub.h"

#include "absl/status/status.h"
#include "cc/ffi/bytes_view.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "program_executor_tee/program_context/cc/noise_client_session.h"
#include "program_executor_tee/program_context/cc/test_helpers.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidentialcompute::outgoing::ComputationDelegation;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::testing::Test;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

constexpr char kWorkerBns[] = "/bns/test/worker";
constexpr int kNumClients = 5;

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return federated_language_jax::CreateXLAExecutor();
}

class NoiseExecutorStubTest : public Test {
 public:
  NoiseExecutorStubTest()
      : fake_computation_delegation_service_({kWorkerBns}, CreateExecutor) {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder computation_delegation_builder;
    computation_delegation_builder.AddListeningPort(
        absl::StrCat(server_address, 0), grpc::InsecureServerCredentials(),
        &port);
    computation_delegation_builder.RegisterService(
        &fake_computation_delegation_service_);
    fake_computation_delegation_server_ =
        computation_delegation_builder.BuildAndStart();
    LOG(INFO) << "ComputationDelegation server listening on "
              << server_address + std::to_string(port) << std::endl;
    computation_delegation_stub_ = ComputationDelegation::NewStub(
        grpc::CreateChannel(absl::StrCat(server_address, port),
                            grpc::InsecureChannelCredentials()));

    // Create a fake SessionConfig for testing.
    auto verifier = bindings::new_fake_attestation_verifier(
        ffi_bindings::BytesView(kFakeEvent),
        ffi_bindings::BytesView(kFakePlatform));
    absl::StatusOr<SessionConfig*> session_config =
        SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                             HandshakeType::kNoiseNN)
            .AddPeerVerifier(kFakeAttesterId, verifier)
            .Build();

    // Create a NoiseExecutorStub wrapping a NoiseClientSession.
    CHECK_OK(session_config.status());
    noise_client_session_ = NoiseClientSession::Create(
        kWorkerBns, session_config.value(), computation_delegation_stub_.get());
    CHECK_OK(noise_client_session_.status());
    noise_executor_stub_ = std::make_unique<NoiseExecutorStub>(
        noise_client_session_.value().get());
  }

  ~NoiseExecutorStubTest() override {
    fake_computation_delegation_server_->Shutdown();
  }

 protected:
  FakeComputationDelegationService fake_computation_delegation_service_;
  std::unique_ptr<Server> fake_computation_delegation_server_;
  std::unique_ptr<ComputationDelegation::Stub> computation_delegation_stub_;
  absl::StatusOr<std::shared_ptr<NoiseClientSession>> noise_client_session_;
  std::unique_ptr<NoiseExecutorStub> noise_executor_stub_;
};

TEST_F(NoiseExecutorStubTest, Success) {
  grpc::ClientContext client_context;

  // Create an executor.
  std::string executor_id;
  {
    tensorflow_federated::v0::GetExecutorResponse get_executor_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->GetExecutor(&client_context,
                                  GetExecutorRequest(kNumClients),
                                  &get_executor_response)
                    .ok());
    executor_id = get_executor_response.executor().id();
    ASSERT_FALSE(executor_id.empty());
  }

  // Create the value 2.
  std::string value_two_ref;
  {
    tensorflow_federated::v0::CreateValueResponse create_value_response_two;
    ASSERT_TRUE(noise_executor_stub_
                    ->CreateValue(&client_context,
                                  CreateIntValueRequest(executor_id, 2),
                                  &create_value_response_two)
                    .ok());
    value_two_ref = create_value_response_two.value_ref().id();
    ASSERT_FALSE(value_two_ref.empty());
  }

  // Create the value 3.
  std::string value_three_ref;
  {
    tensorflow_federated::v0::CreateValueResponse create_value_response_three;
    ASSERT_TRUE(noise_executor_stub_
                    ->CreateValue(&client_context,
                                  CreateIntValueRequest(executor_id, 3),
                                  &create_value_response_three)
                    .ok());
    value_three_ref = create_value_response_three.value_ref().id();
    ASSERT_FALSE(value_three_ref.empty());
  }

  // Create a struct from the values 2 and 3.
  std::string struct_ref;
  {
    tensorflow_federated::v0::CreateStructResponse create_struct_response;
    ASSERT_TRUE(
        noise_executor_stub_
            ->CreateStruct(&client_context,
                           CreateStructRequest(
                               executor_id, {value_two_ref, value_three_ref}),
                           &create_struct_response)
            .ok());
    struct_ref = create_struct_response.value_ref().id();
    ASSERT_FALSE(struct_ref.empty());
  }

  // Create a selection of index 1 from the struct (will be the value 3).
  std::string selection_ref;
  {
    tensorflow_federated::v0::CreateSelectionResponse create_selection_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->CreateSelection(&client_context,
                                      CreateSelectionRequest(
                                          executor_id, struct_ref, /*index=*/1),
                                      &create_selection_response)
                    .ok());
    selection_ref = create_selection_response.value_ref().id();
    ASSERT_FALSE(selection_ref.empty());
  }

  // Materialize the selection.
  {
    tensorflow_federated::v0::ComputeResponse compute_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->Compute(&client_context,
                              ComputeRequest(executor_id, selection_ref),
                              &compute_response)
                    .ok());
    ASSERT_EQ(compute_response.value().array().int32_list().value(0), 3);
  }

  // Create a "federated_value_at_clients" intrinsic.
  std::string intrinsic_comp_ref;
  {
    tensorflow_federated::v0::CreateValueResponse
        create_value_response_intrinsic_comp;
    ASSERT_TRUE(
        noise_executor_stub_
            ->CreateValue(&client_context,
                          CreateIntrinsicValueRequest(
                              executor_id, "federated_value_at_clients"),
                          &create_value_response_intrinsic_comp)
            .ok());
    intrinsic_comp_ref = create_value_response_intrinsic_comp.value_ref().id();
    ASSERT_FALSE(intrinsic_comp_ref.empty());
  }

  // Create a call for the "federated_value_at_clients" intrinsic on the
  // value 2.
  std::string call_ref;
  {
    tensorflow_federated::v0::CreateCallResponse create_call_response;
    ASSERT_TRUE(
        noise_executor_stub_
            ->CreateCall(&client_context,
                         CreateCallRequest(executor_id, intrinsic_comp_ref,
                                           value_two_ref),
                         &create_call_response)
            .ok());
    call_ref = create_call_response.value_ref().id();
    ASSERT_FALSE(call_ref.empty());
  }

  // Materialize the call.
  {
    tensorflow_federated::v0::ComputeResponse compute_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->Compute(&client_context,
                              ComputeRequest(executor_id, call_ref),
                              &compute_response)
                    .ok());
    ASSERT_EQ(
        compute_response.value().federated().type().placement().value().uri(),
        "clients");
    ASSERT_EQ(compute_response.value().federated().value_size(), kNumClients);
  }

  // Dispose all refs.
  {
    tensorflow_federated::v0::DisposeResponse dispose_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->Dispose(&client_context,
                              DisposeRequest(executor_id,
                                             {selection_ref, struct_ref,
                                              value_two_ref, value_three_ref,
                                              intrinsic_comp_ref, call_ref}),
                              &dispose_response)
                    .ok());
  }

  // Dispose the executor.
  {
    tensorflow_federated::v0::DisposeExecutorResponse dispose_executor_response;
    ASSERT_TRUE(noise_executor_stub_
                    ->DisposeExecutor(&client_context,
                                      DisposeExecutorRequest(executor_id),
                                      &dispose_executor_response)
                    .ok());
  }
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee