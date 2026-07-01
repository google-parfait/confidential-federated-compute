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
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "containers/fns/batch_do_fn.h"
#include "containers/fns/fn_factory.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.h"
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
using ::fcp::confidentialcompute::Embedding;
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
//   stream_init: Synthetic embeddings loaded via WriteConfigurationMap
//                → held by factory → passed to constructor
//   Write() (BatchDoFn): Accumulates raw checkpoint blobs
//   Commit() (BatchDoFn): Calls Do() with all accumulated blobs
//   Do(): Parses all checkpoints → computes MAUVE → emits result
class MauveScoreFn : public BatchDoFn {
 public:
  explicit MauveScoreFn(const std::vector<Embedding>& synthetic_data_embeddings)
      : synthetic_data_embeddings_(synthetic_data_embeddings) {}

  absl::Status Do(Any config, std::vector<Session::KV> accumulated_inputs,
                  Context& context) override;

 private:
  const std::vector<Embedding>& synthetic_data_embeddings_;
};

class MauveScoreFnFactory : public FnFactory {
 public:
  explicit MauveScoreFnFactory(std::vector<Embedding> synthetic_data_embeddings)
      : synthetic_data_embeddings_(std::move(synthetic_data_embeddings)) {}

  absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const override {
    return std::make_unique<MauveScoreFn>(synthetic_data_embeddings_);
  }

 private:
  const std::vector<Embedding> synthetic_data_embeddings_;
};

absl::Status MauveScoreFn::Do(Any config,
                              std::vector<Session::KV> accumulated_inputs,
                              Context& context) {
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

  // Phase 5: Serialize the result proto and emit it.
  std::string serialized_result = result.SerializeAsString();
  if (!context.EmitUnencrypted(Session::KV(std::move(serialized_result)))) {
    return absl::InternalError("Failed to emit MAUVE score result.");
  }

  context.GetCounters()["mauve-score-computed"] = 1;
  context.GetCounters()["mauve-real-embeddings-count"] = real_embeddings.size();
  context.GetCounters()["mauve-synth-embeddings-count"] =
      synthetic_data_embeddings_.size();

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
  std::string path = write_configuration_map.at(
      init_config.synthetic_data_embeddings_configuration_id());
  ABSL_ASSIGN_OR_RETURN(std::vector<Embedding> embeddings,
                        read_record_fn(path));
  LOG(INFO) << "Loaded " << embeddings.size() << " synthetic embeddings.";
  return std::make_unique<MauveScoreFnFactory>(std::move(embeddings));
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
