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

#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "cc/oak_session/server_session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "program_executor_tee/program_context/cc/noise_leaf_executor.h"
#include "proto/session/session.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::grpc::ServerContext;
using ::grpc::Status;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

namespace {

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

}  // namespace

FakeComputationDelegationService::FakeComputationDelegationService(
    std::vector<std::string> worker_bns,
    std::function<
        absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
        leaf_executor_factory) {
  for (const auto& worker_bns : worker_bns) {
    auto noise_leaf_executor = NoiseLeafExecutor::Create(
        TestConfigAttestedNNServer, leaf_executor_factory);
    CHECK_OK(noise_leaf_executor);
    noise_leaf_executors_[worker_bns] = std::move(noise_leaf_executor.value());
  }
}

Status FakeComputationDelegationService::Execute(
    ServerContext* context,
    const fcp::confidentialcompute::outgoing::ComputationRequest* request,
    fcp::confidentialcompute::outgoing::ComputationResponse* response) {
  absl::MutexLock lock(&mutex_);
  fcp::confidentialcompute::ComputationRequest worker_request;
  *worker_request.mutable_computation() = request->computation();
  fcp::confidentialcompute::ComputationResponse worker_response;
  grpc::Status execute_status =
      noise_leaf_executors_[request->worker_bns()]->Execute(&worker_request,
                                                            &worker_response);
  if (!execute_status.ok()) {
    return execute_status;
  }
  *response->mutable_result() = std::move(worker_response.result());
  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_executor_tee
