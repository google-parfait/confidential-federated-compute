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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_sql {

// FedSql implementation of Session interface. Not threadsafe.
class FedSqlSession final
    : public confidential_federated_compute::LegacySession {
 public:
  FedSqlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>&
          intrinsics,
      std::optional<SessionInferenceConfiguration> inference_configuration_,
      const std::optional<uint32_t> serialize_output_access_policy_node_id,
      const std::optional<uint32_t> report_output_access_policy_node_id,
      absl::string_view sensitive_values_key);

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
  absl::StatusOr<
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
  ExecuteInferenceAndClientQuery(
      const SqlConfiguration& configuration,
      tensorflow_federated::aggregation::CheckpointParser* parser);

  // The aggregator used during the session to accumulate writes.
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;
  std::optional<const SqlConfiguration> sql_configuration_;
  InferenceModel inference_model_;
  const std::optional<uint32_t> serialize_output_access_policy_node_id_;
  const std::optional<uint32_t> report_output_access_policy_node_id_;
  // Key used to hash sensitive values. In the future we could instead hold an
  // HMAC_CTX to reuse, which might improve performance.
  absl::string_view sensitive_values_key_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_LEDGER_SESSION_H_
