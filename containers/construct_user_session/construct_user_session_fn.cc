// Copyright 2026 Google LLC.
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

#include "containers/construct_user_session/construct_user_session_fn.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "containers/construct_user_session/checkpoint.h"
#include "containers/construct_user_session/row_gather.h"
#include "containers/fns/batch_do_fn.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/util/time_util.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {

namespace {

using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidentialcompute::ConstructUserSessionInitConfig;
using ::google::protobuf::util::TimeUtil;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

// Returns true if any column tensor in `checkpoint` has a dtype that
// conflicts with a previously recorded dtype in `column_dtypes`.
bool HasDtypeConflict(
    const Checkpoint& checkpoint,
    const absl::flat_hash_map<std::string, DataType>& column_dtypes) {
  for (const auto& [name, tensor] : checkpoint.column_tensors()) {
    auto it = column_dtypes.find(name);
    if (it != column_dtypes.end() && it->second != tensor.dtype()) {
      return true;
    }
  }
  return false;
}

// Result of parsing, validating, and grouping all input checkpoints.
struct IngestionResult {
  // Checkpoints grouped by privacy ID.
  absl::flat_hash_map<std::string, std::vector<Checkpoint>> groups;
  // Observed column name -> dtype, used to drive GatherSessionRows.
  absl::flat_hash_map<std::string, DataType> column_dtypes;
};

// Merges input checkpoints that have the same Privacy ID into a single
// output checkpoint, filtering out any rows that are outside the session
// window [window_start, window_end). Input tensors with the same name are
// concatenated into a single output tensor and thus must have the same dtype.
class ConstructUserSessionFn final : public fns::BatchDoFn {
 public:
  explicit ConstructUserSessionFn(absl::Time window_start,
                                  absl::Time window_end,
                                  std::string on_device_query_name)
      : window_start_(window_start),
        window_end_(window_end),
        on_device_query_name_(std::move(on_device_query_name)),
        event_time_tensor_name_(
            absl::StrCat(on_device_query_name_, "/", kEventTimeColumnName)) {}

  absl::Status Do(google::protobuf::Any config,
                  std::vector<Session::KV> accumulated_inputs,
                  Context& context) override;

 private:
  // Parses, deduplicates, validates dtypes, and groups checkpoints by
  // privacy ID.
  IngestionResult IngestAndGroupCheckpoints(
      std::vector<Session::KV> accumulated_inputs, Context& context);

  // Builds serialized session output from `privacy_id_tensor` and
  // `output_tensors`.
  std::string BuildSessionOutput(
      Tensor privacy_id_tensor,
      absl::flat_hash_map<std::string, Tensor> output_tensors);

  // Emits the serialized session output encrypted.
  // Returns a non-OK status only for fatal failures.
  absl::Status EmitSessionOutput(std::string output_data, Context& context);

  const absl::Time window_start_;
  const absl::Time window_end_;
  const std::string on_device_query_name_;
  const std::string event_time_tensor_name_;
};

absl::Status ConstructUserSessionFn::Do(
    google::protobuf::Any config, std::vector<Session::KV> accumulated_inputs,
    Context& context) {
  IngestionResult ingestion_result =
      IngestAndGroupCheckpoints(std::move(accumulated_inputs), context);

  for (auto& [privacy_id, checkpoints] : ingestion_result.groups) {
    // Filter rows against the session window and gather surviving tensors.
    absl::flat_hash_map<std::string, Tensor> output_tensors =
        GatherSessionRows(checkpoints, event_time_tensor_name_, window_start_,
                          window_end_, ingestion_result.column_dtypes);

    // If no rows survive across all input checkpoints, don't bother emitting
    // empty output for this privacy ID.
    if (output_tensors.empty()) {
      continue;
    }

    std::string output_data = BuildSessionOutput(
        checkpoints[0].take_privacy_id_tensor(), std::move(output_tensors));

    // Return if there's a fatal failure that should abort the pipeline.
    ABSL_RETURN_IF_ERROR(EmitSessionOutput(std::move(output_data), context));
  }

  return absl::OkStatus();
}

IngestionResult ConstructUserSessionFn::IngestAndGroupCheckpoints(
    std::vector<Session::KV> accumulated_inputs, Context& context) {
  IngestionResult result;
  absl::flat_hash_set<std::string> seen_blob_ids;

  for (auto& kv : accumulated_inputs) {
    auto [it, inserted] = seen_blob_ids.insert(kv.blob_id);
    if (!inserted) {
      LOG(ERROR) << "Duplicate blob ID encountered: " << kv.blob_id
                 << ". Skipping blob.";
      ++context.GetCounters()["duplicate_blob_count"];
      continue;
    }

    FederatedComputeCheckpointParserFactory parser_factory;
    auto parser = parser_factory.Create(absl::Cord(std::move(kv.data)));
    if (!parser.ok()) {
      LOG(WARNING) << "Skipping blob: failed to create checkpoint parser: "
                   << parser.status();
      ++context.GetCounters()["checkpoint_parse_error_count"];
      continue;
    }
    auto checkpoint = Checkpoint::Create(**parser, on_device_query_name_);
    if (!checkpoint.ok()) {
      LOG(WARNING) << "Skipping blob: failed to create Checkpoint: "
                   << checkpoint.status();
      ++context.GetCounters()["checkpoint_create_error_count"];
      continue;
    }

    // Ignore the whole checkpoint if any tensor dtype conflicts with a
    // previously seen tensor of the same name.
    if (HasDtypeConflict(*checkpoint, result.column_dtypes)) {
      LOG(WARNING) << "Skipping checkpoint: tensor has a dtype mismatch.";
      ++context.GetCounters()["checkpoint_dtype_mismatch_count"];
      continue;
    }
    // Register dtypes now that we know there are no conflicts.
    for (const auto& [name, tensor] : checkpoint->column_tensors()) {
      result.column_dtypes.try_emplace(name, tensor.dtype());
    }

    // Group by privacy ID.
    std::string pid(checkpoint->privacy_id());
    result.groups[std::move(pid)].push_back(*std::move(checkpoint));
  }

  return result;
}

std::string ConstructUserSessionFn::BuildSessionOutput(
    Tensor privacy_id_tensor,
    absl::flat_hash_map<std::string, Tensor> output_tensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  // Add the privacy ID scalar tensor.
  CHECK_OK(builder->Add(fcp::confidential_compute::kPrivacyIdColumnName,
                        std::move(privacy_id_tensor)));

  // Add all output tensors.
  for (auto& [name, tensor] : output_tensors) {
    CHECK_OK(builder->Add(name, std::move(tensor)));
  }

  auto output_data = builder->Build();
  CHECK_OK(output_data.status());
  return std::string(output_data->Flatten());
}

absl::Status ConstructUserSessionFn::EmitSessionOutput(std::string output_data,
                                                       Context& context) {
  // Create the associated metadata with the session window timestamps.
  fcp::confidentialcompute::SessionTimeWindowMetadata time_window_metadata;
  *time_window_metadata.mutable_session_window_start() =
      TimeUtil::NanosecondsToTimestamp(absl::ToUnixNanos(window_start_));
  *time_window_metadata.mutable_session_window_end() =
      TimeUtil::NanosecondsToTimestamp(absl::ToUnixNanos(window_end_));

  fcp::confidentialcompute::AssociatedMetadata associated_metadata;
  associated_metadata.add_metadata()->PackFrom(time_window_metadata);

  Session::KV kv(std::move(output_data));
  kv.associated_metadata = std::move(associated_metadata);

  // Emit the session output encrypted.
  if (!context.EmitEncrypted(/*reencryption_key_index=*/0, std::move(kv))) {
    return absl::InternalError("Failed to emit encrypted session output");
  }
  return absl::OkStatus();
}

// ConstructUserSessionFnFactory holds the parsed configuration and creates
// ConstructUserSessionFn instances.
class ConstructUserSessionFnFactory final : public fns::FnFactory {
 public:
  explicit ConstructUserSessionFnFactory(absl::Time window_start,
                                         absl::Time window_end,
                                         std::string on_device_query_name)
      : window_start_(window_start),
        window_end_(window_end),
        on_device_query_name_(std::move(on_device_query_name)) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<ConstructUserSessionFn>(window_start_, window_end_,
                                                    on_device_query_name_);
  }

 private:
  const absl::Time window_start_;
  const absl::Time window_end_;
  const std::string on_device_query_name_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<fns::FnFactory>>
CreateConstructUserSessionFnFactoryProvider(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const fns::WriteConfigurationMap& write_configuration_map) {
  ConstructUserSessionInitConfig config;
  if (!configuration.UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "ConstructUserSessionInitConfig cannot be unpacked.");
  }

  // Both session_window_start and session_window_end
  // must be explicitly set.
  if (!config.time_window_metadata().has_session_window_start()) {
    return absl::InvalidArgumentError(
        "session_window_start must be explicitly provided.");
  }
  if (!config.time_window_metadata().has_session_window_end()) {
    return absl::InvalidArgumentError(
        "session_window_end must be explicitly provided.");
  }

  absl::Time window_start =
      absl::FromUnixNanos(TimeUtil::TimestampToNanoseconds(
          config.time_window_metadata().session_window_start()));
  absl::Time window_end = absl::FromUnixNanos(TimeUtil::TimestampToNanoseconds(
      config.time_window_metadata().session_window_end()));

  if (window_start >= window_end) {
    return absl::InvalidArgumentError(
        "session_window_start must be strictly before session_window_end.");
  }

  if (config.on_device_query_name().empty()) {
    return absl::InvalidArgumentError(
        "on_device_query_name must be non-empty.");
  }

  return std::make_unique<ConstructUserSessionFnFactory>(
      window_start, window_end, config.on_device_query_name());
}

}  // namespace confidential_federated_compute::construct_user_session
