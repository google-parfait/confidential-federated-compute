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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cc/crypto/signing_key.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/range_tracker.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_sql {

// FedSql implementation of Session interface that works in conjunction with the
// Confidential Federated Compute Key Management Service (CFC KMS). Not
// thread-safe.
class KmsFedSqlSession final
    : public confidential_federated_compute::LegacySession {
 public:
  KmsFedSqlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>&
          intrinsics,
      std::optional<SessionInferenceConfiguration> inference_configuration,
      absl::string_view sensitive_values_key,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash,
      std::shared_ptr<PrivateState> private_state,
      std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle);

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
      const fcp::confidentialcompute::BlobMetadata& unused) override;
  // Accumulates queued blobs into the session state.
  // Returns an error if the aggcore state may be invalid and the session needs
  // to be shut down.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override;

 private:
  // Handles AGGREGATION_TYPE_ACCUMULATE type of SessionWrite.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionAccumulate(
      const fcp::confidentialcompute::BlobMetadata& metadata,
      std::string unencrypted_data);
  // Handles AGGREGATION_TYPE_MERGE type of SessionWrite.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionMerge(
      const fcp::confidentialcompute::BlobMetadata& metadata,
      std::string unencrypted_data);

  // Configuration of the per-client SQL query step.
  struct SqlConfiguration {
    std::string query;
    fcp::confidentialcompute::TableSchema input_schema;
    google::protobuf::RepeatedPtrField<fcp::confidentialcompute::ColumnSchema>
        output_columns;
  };

  // A partially processed but uncommitted input, along with its metadata.
  struct UncommittedInput {
    std::vector<tensorflow_federated::aggregation::Tensor> contents;
    const fcp::confidentialcompute::BlobHeader blob_header;
  };

  // The encrypted intermediate or final result.
  struct EncryptedResult {
    std::string ciphertext;
    fcp::confidentialcompute::BlobMetadata metadata;
    // The configuration for the final result. This is not populated for
    // intermediate results.
    std::optional<fcp::confidentialcompute::FinalResultConfiguration>
        final_result_configuration;
  };

  absl::StatusOr<
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
  ExecuteClientQuery(
      const SqlConfiguration& configuration,
      std::vector<tensorflow_federated::aggregation::Tensor> contents);
  // Encrypts the intermediate result for this session.
  absl::StatusOr<EncryptedResult> EncryptIntermediateResult(
      absl::string_view plaintext);
  // Encrypts the final result for this session.
  absl::StatusOr<EncryptedResult> EncryptFinalResult(
      absl::string_view plaintext);

  // The aggregator used during the session to accumulate writes.
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;
  std::optional<const SqlConfiguration> sql_configuration_;
  InferenceModel inference_model_;
  // Key used to hash sensitive values. In the future we could instead hold an
  // HMAC_CTX to reuse, which might improve performance.
  absl::string_view sensitive_values_key_;
  // The reencryption keys used to re-encrypt the intermediate and final blobs.
  std::vector<std::string> reencryption_keys_;
  // The policy hash used to re-encrypt the intermediate and final blobs with.
  std::string reencryption_policy_hash_;
  // Partially processed uncommitted inputs that will be accumulated the next
  // time SessionCommit is called. Entries are keyed by Blob ID.
  absl::flat_hash_map<absl::uint128, UncommittedInput> uncommitted_inputs_;
  // Tracks committed ranges of blobs for this session.
  RangeTracker range_tracker_;
  // Private state.
  std::shared_ptr<PrivateState> private_state_;
  // The signing key handle used to sign the final result.
  std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_KMS_SESSION_H_
