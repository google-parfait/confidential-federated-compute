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

#include "containers/program_executor_tee/program_context/cc/computation_runner.h"

#include <optional>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "containers/program_executor_tee/program_context/cc/computation_delegation_lambda_runner.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/composing_tee_executor.h"
#include "fcp/confidentialcompute/tee_executor.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidential_compute::CreateComposingTeeExecutor;
using ::fcp::confidential_compute::CreateTeeExecutor;
using ::fcp::confidentialcompute::outgoing::ComputationDelegation;
using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

SessionConfig* GetClientSessionConfig(std::string attester_id) {
  if (attester_id == kFakeAttesterId) {
    auto verifier = bindings::new_fake_attestation_verifier(
        ffi_bindings::BytesView(kFakeEvent),
        ffi_bindings::BytesView(kFakePlatform));
    return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                                HandshakeType::kNoiseNN)
        .AddPeerVerifier(kFakeAttesterId, verifier)
        .Build();
  }
  // TODO: Add config for prod use case.
  return nullptr;
}

std::vector<int> GetNumClientsPerWorker(int num_clients, int num_workers) {
  // Distribute the clients across the child executors. Each child executor
  // should get the same number of clients, with the exception of the last child
  // executor, which may get fewer clients if the number of clients is not
  // evenly divisible by the number of child executors.
  int clients_per_worker =
      std::ceil(static_cast<float>(num_clients) / num_workers);
  std::vector<int> result(num_workers, clients_per_worker);
  int extra_clients = (clients_per_worker * num_workers) - num_clients;
  if (extra_clients > 0) {
    result[num_workers - 1] -= extra_clients;
  }
  return result;
}

// Creates a non-distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>> CreateExecutor(
    int num_clients) {
  auto leaf_executor_fn =
      []() -> absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>> {
    return tensorflow_federated::CreateReferenceResolvingExecutor(
        tensorflow_federated::CreateTensorFlowExecutor());
  };
  tensorflow_federated::CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] = num_clients;
  FCP_ASSIGN_OR_RETURN(auto server_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto client_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto federating_executor,
                       tensorflow_federated::CreateFederatingExecutor(
                           server_child, client_child, cardinality_map));
  return tensorflow_federated::CreateReferenceResolvingExecutor(
      federating_executor);
}

// Executes comp(arg) on the provided TFF execution stack.
absl::StatusOr<tensorflow_federated::v0::Value> Execute(
    std::shared_ptr<tensorflow_federated::Executor> executor,
    tensorflow_federated::v0::Value comp,
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
    std::vector<std::string> worker_bns,
    std::optional<
        std::function<ComputationDelegationResult(ComputationRequest)>>
        computation_delegation_proxy,
    std::string attester_id)
    : worker_bns_(worker_bns), attester_id_(attester_id) {
  if (!worker_bns_.empty()) {
    if (!computation_delegation_proxy.has_value()) {
      LOG(FATAL) << "computation_delegation_proxy must be set when worker_bns "
                    "is not empty.";
    }
    computation_delegation_proxy_ = *computation_delegation_proxy;
    noise_client_sessions_.reserve(worker_bns_.size());
    for (const auto& worker_bns : worker_bns_) {
      // Create a noise client session for each worker and open the session.
      // So the session can be used to send computation requests when the
      // computation is invoked.
      auto client_session = NoiseClientSession::Create(
          worker_bns, GetClientSessionConfig(attester_id),
          std::function(computation_delegation_proxy_));
      CHECK_OK(client_session);
      client_session.value()->OpenSession();
      noise_client_sessions_.push_back(std::move(client_session.value()));
    }
  }
}

// Creates a distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
ComputationRunner::CreateDistributedExecutor(int num_clients) {
  if (worker_bns_.size() < 2) {
    return absl::InvalidArgumentError(
        "worker_bns must have at least 2 entries.");
  }
  // Create a lambda runner for the first worker. This is used to create the
  // server executor.
  auto server_lambda_runner =
      std::make_shared<ComputationDelegationLambdaRunner>(
          noise_client_sessions_[0].get());
  auto server_executor = CreateTeeExecutor(server_lambda_runner, num_clients);
  // Create lambda runners for the remaining workers. These are used to create
  // the child executors.
  int num_workers = worker_bns_.size() - 1;
  std::vector<tensorflow_federated::ComposingChild> composing_children;
  composing_children.reserve(num_workers);
  std::vector<int> num_clients_per_child_worker =
      GetNumClientsPerWorker(num_clients, num_workers);
  for (size_t i = 1; i < worker_bns_.size(); ++i) {
    auto child_lambda_runner =
        std::make_shared<ComputationDelegationLambdaRunner>(
            noise_client_sessions_[i].get());
    int num_clients_current_worker = num_clients_per_child_worker[i - 1];
    auto child_executor =
        CreateTeeExecutor(child_lambda_runner, num_clients_current_worker);
    tensorflow_federated::CardinalityMap cardinality_map;
    cardinality_map[tensorflow_federated::kClientsUri] =
        num_clients_current_worker;
    FCP_ASSIGN_OR_RETURN(auto composing_child,
                         tensorflow_federated::ComposingChild::Make(
                             std::move(child_executor), cardinality_map));
    composing_children.push_back(composing_child);
  }
  auto composing_executor =
      CreateComposingTeeExecutor(server_executor, composing_children);
  return tensorflow_federated::CreateReferenceResolvingExecutor(
      composing_executor);
}

absl::StatusOr<tensorflow_federated::v0::Value> ComputationRunner::InvokeComp(
    int num_clients, const tensorflow_federated::v0::Value comp,
    std::optional<tensorflow_federated::v0::Value> arg) {
  if (worker_bns_.empty()) {
    FCP_ASSIGN_OR_RETURN(auto executor, CreateExecutor(num_clients));
    return Execute(executor, std::move(comp), std::move(arg));
  } else {
    FCP_ASSIGN_OR_RETURN(
        auto executor,
        ComputationRunner::CreateDistributedExecutor(num_clients));
    return Execute(executor, std::move(comp), std::move(arg));
  }
}

}  // namespace confidential_federated_compute::program_executor_tee
