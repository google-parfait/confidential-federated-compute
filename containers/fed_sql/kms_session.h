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

#include "absl/container/flat_hash_set.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fed_sql/dp_unit.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/range_tracker.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "containers/sql/row_set.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_sql {

// Interface for creating new protobuf messages.
class MessageFactory {
 public:
  virtual ~MessageFactory() = default;
  // Creates a new message instance.
  virtual std::unique_ptr<google::protobuf::Message> NewMessage() const = 0;
};

// FedSql implementation of Session interface that works in conjunction with the
// Confidential Federated Compute Key Management Service (CFC KMS). Not
// thread-safe.
class KmsFedSqlSession final : public confidential_federated_compute::Session {
 public:
  KmsFedSqlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>&
          intrinsics,
      std::optional<SessionInferenceConfiguration> inference_configuration,
      std::optional<DpUnitParameters> dp_unit_parameters,
      std::shared_ptr<PrivateState> private_state,
      const absl::flat_hash_set<std::string>& expired_key_ids,
      std::shared_ptr<MessageFactory> message_factory,
      absl::string_view on_device_query_name);

  // Configure the optional per-client SQL query.
  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest configure_request,
      Context& context) override;

  // Incorporates an input into a session. In the case of a client upload,
  // the blob is queued to be eventually processed, but the processing may not
  // finish until SessionCommit is called. In the case of an intermediate
  // partial aggregate, merging with the session state is done
  // promptly without requiring a SessionCommit call.
  //
  // Returns an error if the aggcore state may be invalid and the session
  // needs to be shut down.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override;

  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override;

  // Accumulates queued blobs into the session state.
  // Returns an error if the aggcore state may be invalid and the session needs
  // to be shut down.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override;

 private:
  // Handles AGGREGATION_TYPE_ACCUMULATE type of Write.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Accumulate(
      fcp::confidentialcompute::BlobMetadata metadata,
      std::string unencrypted_data);
  // Handles AGGREGATION_TYPE_MERGE type of Write.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Merge(
      fcp::confidentialcompute::BlobMetadata metadata,
      std::string unencrypted_data);
  // Handles FINALIZATION_TYPE_SERIALIZE type of Finalize.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Serialize(
      Context& context);
  // Handles FINALIZATION_TYPE_PARTITION type of Finalize.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Partition(
      Context& context, uint64_t num_partitions);
  // Handles FINALIZATION_TYPE_REPORT type of Finalize.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Report(
      Context& context);
  // Handles FINALIZATION_TYPE_REPORT_PARTITION type of Finalize.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> ReportPartition(
      Context& context);

  // Produces the final aggregated report and returns the serialized checkpoint.
  absl::StatusOr<absl::Cord> BuildReport();
  // Commits rows, grouping by uncommitted sql::Input. Returns any ignored
  // errors.
  absl::StatusOr<std::vector<absl::Status>> CommitRowsGroupingByInput(
      std::vector<sql::Input>&& uncommitted_inputs,
      const Interval<uint64_t>& range);

  // The aggregator used during the session to accumulate writes.
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;
  std::optional<const SqlConfiguration> sql_configuration_;
  InferenceModel inference_model_;
  // Partially processed uncommitted inputs that will be accumulated the next
  // time SessionCommit is called.
  std::vector<sql::Input> uncommitted_inputs_;
  // The blob IDs of the uncommitted inputs.
  absl::flat_hash_set<absl::uint128> uncommitted_blob_ids_;
  // Tracks committed ranges of blobs for this session.
  RangeTracker range_tracker_;
  // Private state.
  std::shared_ptr<PrivateState> private_state_;
  // Parameters for the differential privacy unit.
  std::optional<DpUnitParameters> dp_unit_parameters_;
  // Prototype for the logged message. Used to create dynamic messages from
  // serialized message checkpoints.
  std::shared_ptr<MessageFactory> message_factory_;
  // The name of the query that executes on the device. This is used to prefix
  // message field-backed column names.
  std::string on_device_query_name_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_KMS_SESSION_H_
