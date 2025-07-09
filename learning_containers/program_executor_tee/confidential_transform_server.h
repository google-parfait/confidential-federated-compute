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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute::program_executor_tee {

// Program executor TEE implementation of Session interface. Not threadsafe.
class ProgramExecutorTeeSession final
    : public confidential_federated_compute::Session {
 public:
  ProgramExecutorTeeSession(
      fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
          initialize_config,
      std::map<std::string, std::string> model_id_to_zip_file,
      confidential_federated_compute::BlobDecryptor* blob_decryptor)
      : initialize_config_(initialize_config),
        model_id_to_zip_file_(model_id_to_zip_file),
        blob_decryptor_(blob_decryptor) {}

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

 private:
  // Initialization config.
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config_;
  // Map of model ids to zip files representing tff FunctionalModels.
  std::map<std::string, std::string> model_id_to_zip_file_;

  confidential_federated_compute::BlobDecryptor* blob_decryptor_;
};

// ConfidentialTransform service for program executor TEE.
class ProgramExecutorTeeConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  ProgramExecutorTeeConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle)
      : ConfidentialTransformBase(std::move(signing_key_handle)) {}

 protected:
  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override;

  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    // TODO: Populate the model_id_to_zip_file_ map from the
    // WriteConfigurationRequests.
    return absl::UnimplementedError(
        "WriteConfigurationRequests are not yet supported.");
  }

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;

 private:
  // Initialization config.
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config_;
  // Map of model ids to zip files representing tff FunctionalModels.
  std::map<std::string, std::string> model_id_to_zip_file_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_
