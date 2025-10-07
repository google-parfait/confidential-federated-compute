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

#include "program_executor_tee/program_context/cc/computation_runner.h"

#include <optional>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/composing_tee_executor.h"
#include "fcp/confidentialcompute/tee_executor.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "program_executor_tee/program_context/cc/noise_executor_stub.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/streaming_remote_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidential_compute::CreateComposingTeeExecutor;
using ::fcp::confidential_compute::CreateTeeExecutor;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::fcp::confidentialcompute::outgoing::ComputationDelegation;
using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::tensorflow_federated::CardinalityMap;
using ::tensorflow_federated::ComposingChild;
using ::tensorflow_federated::CreateFederatingExecutor;
using ::tensorflow_federated::CreateReferenceResolvingExecutor;
using ::tensorflow_federated::CreateStreamingRemoteExecutor;
using ::tensorflow_federated::Executor;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

extern "C" {
extern ::oak::session::SessionConfig* create_session_config(
    const char* reference_values_bytes, size_t reference_values_len);
}

absl::StatusOr<SessionConfig*> GetClientSessionConfig(
    std::string serialized_reference_values) {
  if (serialized_reference_values.empty()) {
    // Create a fake session config for testing.
    auto verifier = bindings::new_fake_attestation_verifier(
        ffi_bindings::BytesView(kFakeEvent),
        ffi_bindings::BytesView(kFakePlatform));
    return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                                HandshakeType::kNoiseNN)
        .AddPeerVerifier(kFakeAttesterId, verifier)
        .Build();
  } else {
    auto* config = create_session_config(serialized_reference_values.data(),
                                         serialized_reference_values.length());
    if (config == nullptr) {
      return absl::InternalError(
          "Failed to create session config from worker reference values.");
    }
    return config;
  }
}

// Creates a non-distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<Executor>> CreateExecutor(
    std::function<absl::StatusOr<std::shared_ptr<Executor>>()>
        leaf_executor_factory,
    int num_clients) {
  auto leaf_executor_fn =
      [leaf_executor_factory]() -> absl::StatusOr<std::shared_ptr<Executor>> {
    FCP_ASSIGN_OR_RETURN(auto executor, leaf_executor_factory());
    return CreateReferenceResolvingExecutor(executor);
  };
  CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] = num_clients;
  FCP_ASSIGN_OR_RETURN(auto server_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto client_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(
      auto federating_executor,
      CreateFederatingExecutor(server_child, client_child, cardinality_map));
  return CreateReferenceResolvingExecutor(federating_executor);
}

// Executes comp(arg) on the provided TFF execution stack.
absl::StatusOr<tensorflow_federated::v0::Value> ExecuteInternal(
    std::shared_ptr<Executor> executor, tensorflow_federated::v0::Value comp,
    std::optional<tensorflow_federated::v0::Value> arg) {
  FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId fn_handle,
                       executor->CreateValue(comp));
  tensorflow_federated::v0::Value call_result;
  if (arg.has_value()) {
    FCP_ASSIGN_OR_RETURN(
        std::shared_ptr<tensorflow_federated::OwnedValueId> arg_handle,
        fcp::confidential_compute::Embed(*arg, executor));
    FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId call_handle,
                         executor->CreateCall(fn_handle, *arg_handle));
    FCP_RETURN_IF_ERROR(executor->Materialize(call_handle, &call_result));
  } else {
    FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId call_handle,
                         executor->CreateCall(fn_handle, std::nullopt));
    FCP_RETURN_IF_ERROR(executor->Materialize(call_handle, &call_result));
  }
  return call_result;
}

}  // namespace

ComputationRunner::ComputationRunner(
    std::function<absl::StatusOr<std::shared_ptr<Executor>>()>
        leaf_executor_factory,
    std::vector<std::string> worker_bns,
    std::string serialized_reference_values,
    std::string outgoing_server_address)
    : leaf_executor_factory_(leaf_executor_factory), worker_bns_(worker_bns) {
  if (!worker_bns_.empty()) {
    stub_ = fcp::confidentialcompute::outgoing::ComputationDelegation::NewStub(
        grpc::CreateChannel(outgoing_server_address,
                            grpc::InsecureChannelCredentials()));
    noise_client_sessions_.reserve(worker_bns_.size());
    for (const auto& worker_bns : worker_bns_) {
      // Create a noise client session for each worker and open the session.
      // So the session can be used to send computation requests when the
      // computation is invoked.
      absl::StatusOr<SessionConfig*> session_config =
          GetClientSessionConfig(serialized_reference_values);
      CHECK_OK(session_config.status());
      auto client_session = NoiseClientSession::Create(
          worker_bns, session_config.value(), stub_.get());
      CHECK_OK(client_session);
      CHECK_OK(client_session.value()->OpenSession());
      noise_client_sessions_.push_back(std::move(client_session.value()));
    }
  }
}

// Creates a distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<Executor>>
ComputationRunner::CreateDistributedExecutor(int num_clients) {
  if (worker_bns_.size() < 2) {
    return absl::InvalidArgumentError(
        "worker_bns must have at least 2 entries.");
  }

  std::vector<ComposingChild> client_executors;
  int remaining_clients = num_clients;
  int num_clients_values_per_executor =
      std::ceil(static_cast<float>(num_clients) / (worker_bns_.size() - 1));
  for (int i = 0; i < worker_bns_.size() - 1; i++) {
    int clients_for_executor =
        std::min(num_clients_values_per_executor, remaining_clients);
    CardinalityMap cardinality_map;
    cardinality_map[tensorflow_federated::kClientsUri] = clients_for_executor;
    client_executors.emplace_back(TFF_TRY(ComposingChild::Make(
        CreateStreamingRemoteExecutor(std::make_unique<NoiseExecutorStub>(
                                          noise_client_sessions_[i].get()),
                                      cardinality_map),
        cardinality_map)));
    remaining_clients -= clients_for_executor;
  }

  CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] = num_clients;
  std::shared_ptr<Executor> server_executor = CreateStreamingRemoteExecutor(
      std::make_unique<NoiseExecutorStub>(noise_client_sessions_.back().get()),
      cardinality_map);

  return CreateReferenceResolvingExecutor(
      CreateComposingExecutor(server_executor, client_executors));
}

grpc::Status ComputationRunner::Execute(
    ::grpc::ServerContext* context,
    const ::fcp::confidentialcompute::outgoing::ComputationRequest* request,
    ::fcp::confidentialcompute::outgoing::ComputationResponse* response) {
  TffSessionConfig session_request;
  if (!request->computation().UnpackTo(&session_request)) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "ComputationRequest cannot be unpacked to TffSessionConfig.");
  }

  // Create executor stack.
  absl::StatusOr<std::shared_ptr<Executor>> executor =
      worker_bns_.empty()
          ? CreateExecutor(leaf_executor_factory_,
                           session_request.num_clients())
          : CreateDistributedExecutor(session_request.num_clients());
  if (!executor.status().ok()) {
    return ToGrpcStatus(executor.status());
  }

  // Execute the computation using the executor stack and return the result.
  absl::StatusOr<tensorflow_federated::v0::Value> result =
      ExecuteInternal(*executor, std::move(session_request.function()),
                      session_request.has_initial_arg()
                          ? std::optional<tensorflow_federated::v0::Value>(
                                session_request.initial_arg())
                          : std::nullopt);
  if (!result.status().ok()) {
    return ToGrpcStatus(result.status());
  }
  response->mutable_result()->PackFrom(*result);
  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_executor_tee
