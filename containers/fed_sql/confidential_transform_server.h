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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/private_state.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "openssl/rand.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

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
      std::shared_ptr<InferenceModel> inference_model =
          std::make_shared<InferenceModel>())
      : ConfidentialTransformBase(crypto_stub,
                                  std::move(encryption_key_handle)),
        inference_model_(inference_model) {
    CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
    std::string key(32, '\0');
    // Generate a random key using BoringSSL. BoringSSL documentation says
    // RAND_bytes always returns 1, so we don't check the return value.
    RAND_bytes(reinterpret_cast<unsigned char*>(key.data()), key.size());
    sensitive_values_key_ = std::move(key);
  };

 private:
  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override;
  absl::Status StreamInitializeTransformWithKms(
      const google::protobuf::Any& configuration,
      const google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys) override;
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
  // These are only required when KMS is enabled for this worker.
  std::optional<std::vector<std::string>> reencryption_keys_;
};

// FedSql implementation of Session interface. Not threadsafe.
class FedSqlSession final : public confidential_federated_compute::Session {
 public:
  FedSqlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>&
          intrinsics,
      std::shared_ptr<InferenceModel> inference_model,
      const std::optional<uint32_t> serialize_output_access_policy_node_id,
      const std::optional<uint32_t> report_output_access_policy_node_id,
      absl::string_view sensitive_values_key)
      : aggregator_(std::move(aggregator)),
        intrinsics_(intrinsics),
        inference_model_(inference_model),
        serialize_output_access_policy_node_id_(
            serialize_output_access_policy_node_id),
        report_output_access_policy_node_id_(
            report_output_access_policy_node_id),
        sensitive_values_key_(sensitive_values_key) {};

  // Configure the optional per-client SQL query.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override;
  // Accumulates a record into the state of the CheckpointAggregator
  // `aggregator`.
  //
  // Returns an error if the aggcore state may be invalid and the session
  // needs to be shut down.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override;
  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override;
  // Currently no action taken for commits.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override {
    return ToSessionCommitResponse(absl::OkStatus());
  }

 private:
  // Configuration of the per-client SQL query step.
  struct SqlConfiguration {
    std::string query;
    fcp::confidentialcompute::TableSchema input_schema;
    google::protobuf::RepeatedPtrField<fcp::confidentialcompute::ColumnSchema>
        output_columns;
  };

  absl::StatusOr<
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
  ExecuteClientQuery(
      const SqlConfiguration& configuration,
      tensorflow_federated::aggregation::CheckpointParser* parser);

  // Session private data (such as privacy budget).
  std::unique_ptr<PrivateState> private_state_;
  // The aggregator used during the session to accumulate writes.
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;
  std::optional<const SqlConfiguration> sql_configuration_;
  std::shared_ptr<InferenceModel> inference_model_;
  const std::optional<uint32_t> serialize_output_access_policy_node_id_;
  const std::optional<uint32_t> report_output_access_policy_node_id_;
  // Key used to hash sensitive values. In the future we could instead hold an
  // HMAC_CTX to reuse, which might improve performance.
  absl::string_view sensitive_values_key_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
