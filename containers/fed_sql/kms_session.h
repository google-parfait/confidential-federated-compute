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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_KMS_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_KMS_SESSION_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "containers/crypto.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/private_state.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "openssl/rand.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_sql {

// FedSql implementation of Session interfacethat works in conjunction with the
// Confidential Federated Compute Key Management Service (CFC KMS). Not
// threadsafe.
class KmsFedSqlSession final : public confidential_federated_compute::Session {
 public:
  KmsFedSqlSession(
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
        sensitive_values_key_(sensitive_values_key) {
    CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
  };

  // Configure the optional per-client SQL query.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override;
  // Incorporates an input into a session. In the case of a client upload,
  // the blob is queued to be eventually processed, but the processing may not
  // finish until SessionCommit is called. In the case of an intermediate
  // partial aggregate, merging with the session state is done
  // promptly without requiring a SessionCommit call.
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
  // Accumulates queued blobs into the session state.
  // Returns an error if the aggcore state may be invalid and the session needs
  // to be shut down.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override;

 private:
  // Configuration of the per-client SQL query step.
  struct SqlConfiguration {
    std::string query;
    fcp::confidentialcompute::TableSchema input_schema;
    google::protobuf::RepeatedPtrField<fcp::confidentialcompute::ColumnSchema>
        output_columns;
  };

  // A partially processed but uncommitted input, along with its metadata.
  struct UncommittedInput {
    std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser> parser;
    const fcp::confidentialcompute::BlobMetadata metadata;
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
  // SQL query results that will be accumulated the next time SessionCommit is
  // called.
  std::vector<UncommittedInput> uncommitted_inputs_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_KMS_SESSION_H_
