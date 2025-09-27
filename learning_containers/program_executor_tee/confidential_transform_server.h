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

#include <pybind11/functional.h>

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
      confidential_federated_compute::BlobDecryptor* blob_decryptor,
      std::function<std::optional<pybind11::function>()>
          get_program_initialize_fn)
      : initialize_config_(initialize_config),
        model_id_to_zip_file_(model_id_to_zip_file),
        blob_decryptor_(blob_decryptor),
        get_program_initialize_fn_(get_program_initialize_fn) {}

  // Configures a minimal session.
  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest request,
      Context& context) override;

  // Not supported.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest request,
      std::string unencrypted_data, Context& context) override;

  // Not supported.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest request,
      Context& context) override;

  // Triggers program execution.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override;

 private:
  // Initialization config.
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config_;

  // Map of model ids to zip files representing tff FunctionalModels.
  std::map<std::string, std::string> model_id_to_zip_file_;

  // Blob decryptor.
  confidential_federated_compute::BlobDecryptor* blob_decryptor_;

  // Function that generates the optional pybind function to call at the
  // beginning of the program runner.
  std::function<std::optional<pybind11::function>()> get_program_initialize_fn_;
};

// ConfidentialTransform service for program executor TEE.
class ProgramExecutorTeeConfidentialTransform
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
          write_configuration) override;

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;

  virtual std::optional<pybind11::function> GetProgramInitializeFn() {
    return std::nullopt;
  }

  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
  GetInitializeConfig() {
    return initialize_config_;
  }

  std::vector<std::string> GetWorkerBnsAddresses() {
    return worker_bns_addresses_;
  }

 private:
  // Initialization config.
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config_;

  // List of worker bns addresses.
  std::vector<std::string> worker_bns_addresses_;

  // Map of model ids to zip files representing tff FunctionalModels.
  std::map<std::string, std::string> model_id_to_zip_file_;

  // Track the model_id of the current model passed to container through
  // `ReadWriteConfigurationRequest`.
  std::string current_model_id_;

  // Tracking zipped models passed into the container through
  // WriteConfigurationRequest.
  struct WriteConfigurationMetadata {
    std::string file_path;
    uint64_t total_size_bytes;
    bool commit;
  };
  absl::flat_hash_map<std::string, WriteConfigurationMetadata>
      write_configuration_map_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_H_
