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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// TODO: Determine the port of the DataReadWrite service from the config
// provided at initialization time, instead of using a hardcoded port.
inline constexpr int kDataReadWriteServicePort = 500051;

// Program executor TEE implementation of Session interface. Not threadsafe.
class ProgramExecutorTeeSession final
    : public confidential_federated_compute::Session {
 public:
  ProgramExecutorTeeSession() {};

  // Configures a minimal session.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override;
  // Not supported.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override;
  // Currently no action taken for commits.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override {
    return ToSessionCommitResponse(absl::OkStatus());
  }
  // Triggers program execution.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override;
};

// ConfidentialTransform service for program executor TEE.
class ProgramExecutorTeeConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  ProgramExecutorTeeConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
      : ConfidentialTransformBase(crypto_stub) {};

 protected:
  virtual absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override;
  virtual absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }

  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    return std::make_unique<ProgramExecutorTeeSession>();
  }
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_
