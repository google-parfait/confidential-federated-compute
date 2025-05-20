// Copyright 2024 Google LLC.
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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/private_state.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/access_policy.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"

namespace confidential_federated_compute::fed_sql {

// ConfidentialTransform service for Federated SQL. Executes the aggregation
// step of FedSQL.
// TODO: execute the per-client SQL query step.
class FedSqlConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  FedSqlConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub,
      std::unique_ptr<::oak::crypto::EncryptionKeyHandle>
          encryption_key_handle = nullptr,
      std::shared_ptr<::oak::crypto::SigningKeyHandle> signing_key_handle =
          nullptr,
      std::shared_ptr<InferenceModel> inference_model =
          std::make_shared<InferenceModel>());

 private:
  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override;
  absl::Status StreamInitializeTransformWithKms(
      const google::protobuf::Any& configuration,
      const google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash) override;
  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override;
  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;

  //  Set the intrinsics based on the initialization configuration.
  absl::Status SetAndValidateIntrinsics(
      const fcp::confidentialcompute::FedSqlContainerInitializeConfiguration&
          config);
  // Returns the configuration constraints for this worker. These must be
  // validated by the Ledger. This function is not used when KMS is enabled for
  // this worker.
  absl::StatusOr<google::protobuf::Struct> GetConfigConstraints(
      const fcp::confidentialcompute::FedSqlContainerInitializeConfiguration&
          config);
  // Validates the configuration constraints received from KMS.
  absl::Status ValidateConfigConstraints(
      const fcp::confidentialcompute::FedSqlContainerConfigConstraints&
          config_constraints);
  // Initialized the private state - the initial budget received from KMS.
  absl::Status InitializePrivateState(
      const fcp::confidentialcompute::AccessBudget& access_budget);
  // Initialize the inference model with the given configuration.
  absl::Status InitializeInferenceModel(
      const fcp::confidentialcompute::InferenceInitializeConfiguration&
          inference_init_config);

  absl::Mutex mutex_;
  std::optional<const std::vector<tensorflow_federated::aggregation::Intrinsic>>
      intrinsics_ ABSL_GUARDED_BY(mutex_);
  std::optional<uint32_t> serialize_output_access_policy_node_id_;
  std::optional<uint32_t> report_output_access_policy_node_id_;
  // Key used to hash sensitive values. Once we start partitioning the join
  // data, we likely want this to be held by the FedSqlSession instead.
  std::string sensitive_values_key_;
  std::optional<SessionInferenceConfiguration> inference_configuration_;
  std::shared_ptr<InferenceModel> inference_model_;
  // Track the configuration ID of the current data blob passed to container
  // through `ReadWriteConfigurationRequest`.
  std::string current_configuration_id_;
  // Tracking data passed into the container through WriteConfigurationRequest.
  struct WriteConfigurationMetadata {
    std::string file_path;
    uint64_t total_size_bytes;
    bool commit;
  };
  absl::flat_hash_map<std::string, WriteConfigurationMetadata>
      write_configuration_map_;
  // Reencryption keys for the resultant outputs.
  // The below fields are only set when KMS is enabled for this worker.
  std::optional<std::vector<std::string>> reencryption_keys_;
  // The policy hash used to re-encrypt the intermediate and final blobs with.
  std::optional<std::string> reencryption_policy_hash_;
  // Initial private state shared between all sessions.
  std::shared_ptr<PrivateState> private_state_;
  // The signing key handle used to sign the final results.
  // This is only required when KMS is enabled for this worker.
  std::shared_ptr<::oak::crypto::SigningKeyHandle> signing_key_handle_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
