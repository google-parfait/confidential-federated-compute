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
#include "containers/fns/batch_do_fn.h"
#include "containers/fns/fn_factory.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.h"
#include "mauve_budget_state.pb.h"
#include "py_mauve_delegate.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "utils.h"

namespace confidential_federated_compute::mauve_score {

using ReadRecordFn = absl::AnyInvocable<absl::StatusOr<
    std::vector<fcp::confidentialcompute::Embedding>>(absl::string_view)>;

namespace {

constexpr absl::string_view kDataTensorName = "data";

using ::confidential_federated_compute::fns::BatchDoFn;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::MauveScoreContainerConfigConstraints;
using ::fcp::confidentialcompute::MauveScoreContainerInitializeConfiguration;
using ::fcp::confidentialcompute::MauveScoreResult;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

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
//         computes MAUVE → stores result for FinalizeReplica
//   FinalizeReplica(): Emits the stored result via EmitReleasable with
//                      budget state tracking
class MauveScoreFn : public BatchDoFn {
 public:
  static absl::StatusOr<std::unique_ptr<MauveScoreFn>> Create(
      const std::vector<Embedding>& synthetic_data_embeddings,
      uint32_t access_budget_times,
      std::optional<std::string> initial_pipeline_state);

  absl::Status Do(Any config, std::vector<Session::KV> accumulated_inputs,
                  DoContext& context) override;

  absl::Status FinalizeReplica(Any config, FnContext& context) override;

 private:
  MauveScoreFn(const std::vector<Embedding>& synthetic_data_embeddings,
               Budget budget)
      : synthetic_data_embeddings_(synthetic_data_embeddings),
        budget_(std::move(budget)) {};
  const std::vector<Embedding>& synthetic_data_embeddings_;
  Budget budget_;
  // Stored result from Do(), to be emitted in FinalizeReplica().
  std::string serialized_result_;
};

class MauveScoreFnFactory : public FnFactory {
 public:
  MauveScoreFnFactory(std::vector<Embedding> synthetic_data_embeddings,
                      uint32_t access_budget_times,
                      std::optional<std::string> initial_pipeline_state)
      : synthetic_data_embeddings_(std::move(synthetic_data_embeddings)),
        access_budget_times_(access_budget_times),
        initial_pipeline_state_(std::move(initial_pipeline_state)) {}

  absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const override {
    return MauveScoreFn::Create(synthetic_data_embeddings_,
                                access_budget_times_, initial_pipeline_state_);
  }

 private:
  const std::vector<Embedding> synthetic_data_embeddings_;
  const uint32_t access_budget_times_;
  std::optional<std::string> initial_pipeline_state_;
};

absl::StatusOr<std::unique_ptr<MauveScoreFn>> MauveScoreFn::Create(
    const std::vector<Embedding>& synthetic_data_embeddings,
    uint32_t access_budget_times,
    std::optional<std::string> initial_pipeline_state) {
  ABSL_ASSIGN_OR_RETURN(
      Budget budget,
      Budget::Create(std::move(initial_pipeline_state), access_budget_times));
  return absl::WrapUnique(
      new MauveScoreFn(synthetic_data_embeddings, std::move(budget)));
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

  // Phase 1: Parse all accumulated checkpoint blobs into flat float vectors.
  std::vector<std::vector<float>> real_embeddings;
  for (auto& kv : accumulated_inputs) {
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

  // Phase 2: Validate inputs.
  if (real_embeddings.empty()) {
    return absl::InvalidArgumentError("No real embeddings received.");
  }
  if (synthetic_data_embeddings_.empty()) {
    return absl::InvalidArgumentError("No synthetic embeddings loaded.");
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

  // Phase 4: Compute MAUVE score via Python (pybind11).
  ABSL_ASSIGN_OR_RETURN(
      MauveScoreResult result,
      ComputeMauveViaPython(real_embeddings, synth_embeddings));

  LOG(INFO) << "MAUVE AUC: " << result.mauve_auc()
            << ", clusters: " << result.num_clusters()
            << ", recall: " << result.recall()
            << ", precision: " << result.precision();

  // Phase 5: Store the serialized result for FinalizeReplica.
  serialized_result_ = result.SerializeAsString();

  // Decrement budget
  budget_.DecrementBudget();

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
    ReadRecordFn read_record_fn) {
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

  std::string path = write_configuration_map.at(
      init_config.synthetic_data_embeddings_configuration_id());
  ABSL_ASSIGN_OR_RETURN(std::vector<Embedding> embeddings,
                        read_record_fn(path));
  LOG(INFO) << "Loaded " << embeddings.size() << " synthetic embeddings.";

  return std::make_unique<MauveScoreFnFactory>(
      std::move(embeddings), access_budget_times, std::move(initial_state));
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
