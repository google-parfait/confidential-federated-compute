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
#include "mauve_score_fn.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "budget.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/fns/batch_do_fn.h"
#include "containers/fns/fn_factory.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.h"
#include "py_mauve_delegate.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "utils.h"

namespace confidential_federated_compute::mauve_score {

using ReadRecordFn = absl::AnyInvocable<absl::StatusOr<
    std::vector<fcp::confidentialcompute::Embedding>>(absl::string_view)>;

using ComputeMauveFn =
    std::function<absl::StatusOr<fcp::confidentialcompute::MauveScoreResult>(
        const std::vector<std::vector<float>>&,
        const std::vector<std::vector<float>>&)>;

namespace {

constexpr absl::string_view kDataTensorName = "data";

using ::confidential_federated_compute::fns::BatchDoFn;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::MauveScoreContainerConfigConstraints;
using ::fcp::confidentialcompute::MauveScoreContainerInitializeConfiguration;
using ::fcp::confidentialcompute::MauveScoreResult;
using ::fcp::confidentialcompute::SessionTimeWindowMetadata;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

// Key used for budget tracking, either a time interval or a set of active key
// ids.
struct BudgetKey {
  absl::flat_hash_set<std::string> active_keys;
  std::optional<Interval<uint64_t>> agg_window;
};

bool HasTimeWindowMetadata(const Session::KV& kv) {
  if (!kv.associated_metadata.has_value()) {
    return false;
  }
  for (const auto& entry : kv.associated_metadata->metadata()) {
    if (entry.Is<SessionTimeWindowMetadata>()) {
      return true;
    }
  }
  return false;
}

// MauveScoreFn extends BatchDoFn to compute the MAUVE score over
// accumulated real embeddings.
//
// Data flow:
//   stream_init: Synthetic embeddings & initial pipeline state loaded via
//                WriteConfigurationMap → held by factory → passed to
//                constructor
//   Write() (BatchDoFn): Accumulates raw checkpoint blobs
//   Commit() (BatchDoFn): Calls Do() with all accumulated blobs
//   Do(): Parses all checkpoints → checks for duplicate blob IDs →
//         validates and updates budget → computes MAUVE → stores result for
//         FinalizeReplica
//   FinalizeReplica(): Emits the stored result via EmitReleasable with
//                      budget state tracking
class MauveScoreFn : public BatchDoFn {
 public:
  static absl::StatusOr<std::unique_ptr<MauveScoreFn>> Create(
      const std::vector<Embedding>& synthetic_data_embeddings,
      uint32_t access_budget_times, uint64_t min_agg_window_minutes,
      std::optional<std::string> initial_pipeline_state,
      ComputeMauveFn compute_mauve_fn);

  absl::Status Do(Any config, std::vector<Session::KV> accumulated_inputs,
                  DoContext& context) override;

  absl::Status FinalizeReplica(Any config, FnContext& context) override;

 private:
  MauveScoreFn(const std::vector<Embedding>& synthetic_data_embeddings,
               Budget budget, uint64_t min_agg_window_minutes,
               ComputeMauveFn compute_mauve_fn)
      : synthetic_data_embeddings_(synthetic_data_embeddings),
        budget_(std::move(budget)),
        min_agg_window_minutes_(min_agg_window_minutes),
        compute_mauve_fn_(std::move(compute_mauve_fn)) {}

  absl::StatusOr<std::vector<Session::KV>>
  FilterInputsWithRemainingTimeWindowBudget(
      std::vector<Session::KV> accumulated_inputs, DoContext& context);
  absl::StatusOr<std::vector<Session::KV>> FilterInputsWithRemainingKeyIdBudget(
      std::vector<Session::KV> accumulated_inputs, DoContext& context);

  const std::vector<Embedding>& synthetic_data_embeddings_;
  Budget budget_;
  const uint64_t min_agg_window_minutes_;
  ComputeMauveFn compute_mauve_fn_;
  // Stored result from Do(), to be emitted in FinalizeReplica().
  std::string serialized_result_;
  BudgetKey budget_key_;
};

class MauveScoreFnFactory : public FnFactory {
 public:
  MauveScoreFnFactory(std::vector<Embedding> synthetic_data_embeddings,
                      uint32_t access_budget_times,
                      uint64_t min_agg_window_minutes,
                      std::optional<std::string> initial_pipeline_state,
                      ComputeMauveFn compute_mauve_fn)
      : synthetic_data_embeddings_(std::move(synthetic_data_embeddings)),
        access_budget_times_(access_budget_times),
        min_agg_window_minutes_(min_agg_window_minutes),
        initial_pipeline_state_(std::move(initial_pipeline_state)),
        compute_mauve_fn_(std::move(compute_mauve_fn)) {}

  absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const override {
    return MauveScoreFn::Create(synthetic_data_embeddings_,
                                access_budget_times_, min_agg_window_minutes_,
                                initial_pipeline_state_, compute_mauve_fn_);
  }

 private:
  const std::vector<Embedding> synthetic_data_embeddings_;
  const uint32_t access_budget_times_;
  const uint64_t min_agg_window_minutes_;
  std::optional<std::string> initial_pipeline_state_;
  ComputeMauveFn compute_mauve_fn_;
};

absl::StatusOr<std::unique_ptr<MauveScoreFn>> MauveScoreFn::Create(
    const std::vector<Embedding>& synthetic_data_embeddings,
    uint32_t access_budget_times, uint64_t min_agg_window_minutes,
    std::optional<std::string> initial_pipeline_state,
    ComputeMauveFn compute_mauve_fn) {
  ABSL_ASSIGN_OR_RETURN(
      Budget budget,
      Budget::Create(std::move(initial_pipeline_state), access_budget_times));
  return absl::WrapUnique(
      new MauveScoreFn(synthetic_data_embeddings, std::move(budget),
                       min_agg_window_minutes, std::move(compute_mauve_fn)));
}

absl::StatusOr<std::vector<Session::KV>>
MauveScoreFn::FilterInputsWithRemainingTimeWindowBudget(
    std::vector<Session::KV> accumulated_inputs, DoContext& context) {
  std::optional<Interval<uint64_t>> agg_window;
  std::vector<Session::KV> valid_inputs;

  for (auto& kv : accumulated_inputs) {
    SessionTimeWindowMetadata time_window_metadata;
    bool found = false;
    if (kv.associated_metadata.has_value()) {
      for (const auto& entry : kv.associated_metadata->metadata()) {
        if (entry.UnpackTo(&time_window_metadata)) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      return absl::InvalidArgumentError(
          "Missing SessionTimeWindowMetadata in time-window budget mode.");
    }

    Interval<uint64_t> window(
        time_window_metadata.session_window_start().seconds(),
        time_window_metadata.session_window_end().seconds());
    bool has_budget = budget_.HasRemainingBudget(window);
    for (const auto& kid : time_window_metadata.key_ids()) {
      if (!budget_.HasRemainingBudget(kid)) {
        has_budget = false;
        break;
      }
    }
    if (!has_budget) {
      context.IncrementCounter("mauve-ignored-exhausted-budget-blobs-count");
      continue;
    }

    agg_window =
        agg_window.has_value()
            ? Interval<uint64_t>(std::min(agg_window->start(), window.start()),
                                 std::max(agg_window->end(), window.end()))
            : window;
    valid_inputs.push_back(std::move(kv));
  }

  if (valid_inputs.empty()) {
    return absl::FailedPreconditionError(
        "No real embeddings remaining after filtering blobs with exhausted "
        "budget.");
  }

  uint64_t window_duration_minutes =
      (agg_window->end() - agg_window->start()) / 60;
  if (window_duration_minutes < min_agg_window_minutes_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The aggregation window duration (", window_duration_minutes,
        " minutes) is less than the minimum required (",
        min_agg_window_minutes_, " minutes)."));
  }

  budget_key_.agg_window = agg_window;
  return valid_inputs;
}

absl::StatusOr<std::vector<Session::KV>>
MauveScoreFn::FilterInputsWithRemainingKeyIdBudget(
    std::vector<Session::KV> accumulated_inputs, DoContext& context) {
  absl::flat_hash_set<std::string> active_keys;
  std::vector<Session::KV> valid_inputs;

  for (auto& kv : accumulated_inputs) {
    std::string key_id = "";
    if (kv.associated_metadata.has_value()) {
      for (const auto& entry : kv.associated_metadata->metadata()) {
        BlobHeader blob_header;
        if (entry.UnpackTo(&blob_header)) {
          key_id = blob_header.key_id();
          break;
        }
      }
    }

    if (!budget_.HasRemainingBudget(key_id)) {
      context.IncrementCounter("mauve-ignored-exhausted-budget-blobs-count");
      continue;
    }

    active_keys.insert(key_id);
    valid_inputs.push_back(std::move(kv));
  }

  if (valid_inputs.empty()) {
    return absl::FailedPreconditionError(
        "No real embeddings remaining after filtering blobs with exhausted "
        "budget.");
  }

  budget_key_.active_keys = std::move(active_keys);
  return valid_inputs;
}

absl::Status MauveScoreFn::Do(Any config,
                              std::vector<Session::KV> accumulated_inputs,
                              DoContext& context) {
  // Phase 0: Check for duplicate blob IDs.
  absl::flat_hash_set<std::string> seen_blob_ids;
  for (const auto& kv : accumulated_inputs) {
    if (!kv.blob_id.empty()) {
      auto [it, inserted] = seen_blob_ids.insert(kv.blob_id);
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate blob id detected: ", kv.blob_id));
      }
    }
  }

  if (accumulated_inputs.empty()) {
    return absl::InvalidArgumentError("No real embeddings received.");
  }
  if (synthetic_data_embeddings_.empty()) {
    return absl::InvalidArgumentError("No synthetic embeddings loaded.");
  }

  // Phase 1: Filter blobs by budget and update budget state.
  std::vector<Session::KV> valid_inputs;
  if (HasTimeWindowMetadata(accumulated_inputs.front())) {
    ABSL_ASSIGN_OR_RETURN(valid_inputs,
                          FilterInputsWithRemainingTimeWindowBudget(
                              std::move(accumulated_inputs), context));
  } else {
    ABSL_ASSIGN_OR_RETURN(valid_inputs,
                          FilterInputsWithRemainingKeyIdBudget(
                              std::move(accumulated_inputs), context));
  }

  // Phase 2: Parse valid checkpoint blobs into flat float vectors.
  std::vector<std::vector<float>> real_embeddings;
  for (auto& kv : valid_inputs) {
    FederatedComputeCheckpointParserFactory parser_factory;
    ABSL_ASSIGN_OR_RETURN(
        auto parser, parser_factory.Create(absl::Cord(std::move(kv.data))));
    ABSL_ASSIGN_OR_RETURN(auto tensor,
                          parser->GetTensor(std::string(kDataTensorName)));
    if (tensor.dtype() != DT_FLOAT) {
      return absl::InvalidArgumentError(
          "The input tensor is not a float tensor.");
    }
    auto dims = tensor.shape().dim_sizes();
    if (dims.size() != 2) {
      return absl::InvalidArgumentError(
          "The input tensor is not a two-dimensional tensor.");
    }
    int32_t batch_dim = dims[0];
    int32_t emb_dim = dims[1];
    absl::Span<const float> data = tensor.AsSpan<float>();

    for (int i = 0; i < batch_dim; i++) {
      auto emb_span = data.subspan(i * emb_dim, emb_dim);
      real_embeddings.emplace_back(emb_span.begin(), emb_span.end());
    }
  }

  LOG(INFO) << "Computing MAUVE score with " << real_embeddings.size()
            << " real and " << synthetic_data_embeddings_.size()
            << " synthetic embeddings.";

  // Phase 3: Convert synthetic Embedding protos to flat float vectors.
  std::vector<std::vector<float>> synth_embeddings;
  synth_embeddings.reserve(synthetic_data_embeddings_.size());
  for (const auto& emb : synthetic_data_embeddings_) {
    synth_embeddings.emplace_back(emb.values().begin(), emb.values().end());
  }

  // Phase 4: Compute MAUVE score.
  ABSL_ASSIGN_OR_RETURN(MauveScoreResult result,
                        compute_mauve_fn_(real_embeddings, synth_embeddings));

  LOG(INFO) << "MAUVE AUC: " << result.mauve_auc()
            << ", clusters: " << result.num_clusters()
            << ", recall: " << result.recall()
            << ", precision: " << result.precision();

  // Phase 5: Store the serialized result for FinalizeReplica.
  serialized_result_ = result.SerializeAsString();

  context.IncrementCounter("mauve-score-computed");
  context.IncrementCounterBy("mauve-real-embeddings-count",
                             real_embeddings.size());
  context.IncrementCounterBy("mauve-synth-embeddings-count",
                             synthetic_data_embeddings_.size());

  return absl::OkStatus();
}

absl::Status MauveScoreFn::FinalizeReplica(Any config, FnContext& context) {
  if (serialized_result_.empty()) {
    return absl::FailedPreconditionError(
        "No MAUVE result available. Was Do() called successfully?");
  }

  if (budget_key_.agg_window.has_value()) {
    ABSL_RETURN_IF_ERROR(budget_.UpdateTimeBudget(*budget_key_.agg_window));
  } else {
    ABSL_RETURN_IF_ERROR(budget_.UpdatePerKeyBudget(budget_key_.active_keys));
  }

  // Compute the destination state.
  std::string dst_state = budget_.SerializeAsString();

  // Emit the result via EmitReleasable.
  if (!context.EmitReleasable(/*reencryption_key_index=*/0,
                              Session::KV(std::move(serialized_result_)),
                              budget_.GetInitialState(), dst_state)) {
    return absl::InternalError("Failed to emit MAUVE score result.");
  }

  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<std::unique_ptr<FnFactory>> ProvideMauveScoreFnFactory(
    const Any& configuration, const Any& config_constraints,
    const WriteConfigurationMap& write_configuration_map,
    ReadRecordFn read_record_fn, ComputeMauveFn compute_mauve_fn = nullptr) {
  if (!compute_mauve_fn) {
    compute_mauve_fn = ComputeMauveViaPython;
  }
  MauveScoreContainerInitializeConfiguration init_config;
  if (!configuration.UnpackTo(&init_config)) {
    return absl::InvalidArgumentError(
        "Cannot unpack init config to "
        "MauveScoreContainerInitializeConfiguration.");
  }
  if (!write_configuration_map.contains(
          init_config.synthetic_data_embeddings_configuration_id())) {
    return absl::InvalidArgumentError(
        "Write configuration map doesn't contain synthetic data embeddings "
        "configuration id.");
  }

  // Read the initial pipeline state from write_configuration_map.
  auto state_it = write_configuration_map.find(kPrivateStateConfigId);
  if (state_it == write_configuration_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected '", kPrivateStateConfigId,
                     "' configuration id is not found."));
  }
  const std::string& private_state_path = state_it->second;
  std::ifstream private_state_file(private_state_path);
  if (!private_state_file.is_open()) {
    return absl::DataLossError(
        absl::StrCat("Failed to open file for reading: ", private_state_path));
  }
  auto state_size = std::filesystem::file_size(private_state_path);
  std::optional<std::string> initial_state = std::nullopt;
  if (state_size > 0) {
    std::string initial_pipeline_state(state_size, '\0');
    private_state_file.read(initial_pipeline_state.data(), state_size);
    initial_state = std::move(initial_pipeline_state);
  }

  // Parse the config constraints to get the access budget.
  MauveScoreContainerConfigConstraints mauve_constraints;
  if (!config_constraints.UnpackTo(&mauve_constraints)) {
    return absl::InvalidArgumentError(
        "Cannot unpack config_constraints to "
        "MauveScoreContainerConfigConstraints.");
  }
  if (!mauve_constraints.has_access_budget() ||
      !mauve_constraints.access_budget().has_times()) {
    return absl::InvalidArgumentError(
        "Config constraints must specify an access budget with times.");
  }
  uint32_t access_budget_times = mauve_constraints.access_budget().times();
  if (access_budget_times <= 0) {
    return absl::InvalidArgumentError(
        "Access budget must be greater than zero.");
  }
  uint64_t min_agg_window_minutes = mauve_constraints.min_agg_window_minutes();

  std::string path = write_configuration_map.at(
      init_config.synthetic_data_embeddings_configuration_id());
  ABSL_ASSIGN_OR_RETURN(std::vector<Embedding> embeddings,
                        read_record_fn(path));
  LOG(INFO) << "Loaded " << embeddings.size() << " synthetic embeddings.";

  return std::make_unique<MauveScoreFnFactory>(
      std::move(embeddings), access_budget_times, min_agg_window_minutes,
      std::move(initial_state), std::move(compute_mauve_fn));
}

fns::FnFactoryProvider CreateMauveScoreFnFactoryProvider() {
  return [](const Any& configuration, const Any& config_constraints,
            const WriteConfigurationMap& write_configuration_map)
             -> absl::StatusOr<std::unique_ptr<FnFactory>> {
    return ProvideMauveScoreFnFactory(configuration, config_constraints,
                                      write_configuration_map, ReadRecords);
  };
}

}  // namespace confidential_federated_compute::mauve_score
