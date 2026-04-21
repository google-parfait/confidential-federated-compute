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
#include "nn_histogram_fn.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fns/do_fn.h"
#include "containers/fns/fn_factory.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/nn_histogram_config.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "utils.h"

namespace confidential_federated_compute::nn_histogram {
namespace {

using ::confidential_federated_compute::fns::DoFn;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::NNHistogramContainerInitializeConfiguration;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

class NNHistogramFn : public DoFn {
 public:
  explicit NNHistogramFn(
      const std::vector<Embedding>& synthetic_data_embeddings,
      std::shared_ptr<NearestNeighborFn> nn_fn)
      : synthetic_data_embeddings_(synthetic_data_embeddings), nn_fn_(nn_fn) {}

  absl::Status Do(KV input, Context& context) override;

 private:
  const std::vector<Embedding>& synthetic_data_embeddings_;
  std::shared_ptr<NearestNeighborFn> nn_fn_;
};

class NNHistogramFnFactory : public FnFactory {
 public:
  explicit NNHistogramFnFactory(
      std::vector<Embedding> synthetic_data_embeddings, NearestNeighborFn nn_fn)
      : synthetic_data_embeddings_(std::move(synthetic_data_embeddings)),
        nn_fn_(std::make_shared<NearestNeighborFn>(std::move(nn_fn))) {}
  absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const override {
    return std::make_unique<NNHistogramFn>(synthetic_data_embeddings_, nn_fn_);
  }

 private:
  // Holds the ownership of synthetic data embeddings.
  const std::vector<Embedding> synthetic_data_embeddings_;
  std::shared_ptr<NearestNeighborFn> nn_fn_;
};

absl::Status NNHistogramFn::Do(KV input, Context& context) {
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(
      auto parser, parser_factory.Create(absl::Cord(std::move(input.data))));
  FCP_ASSIGN_OR_RETURN(auto tensor,
                       parser->GetTensor(std::string(kDataTensorName)));
  if (tensor.dtype() != DT_FLOAT) {
    return absl::InvalidArgumentError("The input tensor not a float tensor.");
  }
  auto dims = tensor.shape().dim_sizes();
  if (dims.size() != 2) {
    return absl::InvalidArgumentError(
        "The input tensor is not a two-dimension tensor.");
  }
  int32_t batch_dim = dims[0];
  int32_t emb_dim = dims[1];
  absl::Span<const float> data = tensor.AsSpan<float>();
  std::vector<int32_t> nn_indices;
  nn_indices.reserve(batch_dim);
  for (int i = 0; i < batch_dim; i++) {
    auto input_emb = data.subspan(i * emb_dim, emb_dim);
    FCP_ASSIGN_OR_RETURN(int32_t index,
                         (*nn_fn_)(input_emb, synthetic_data_embeddings_));
    nn_indices.push_back(index);
  }

  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  Tensor index_t(std::move(nn_indices));
  FCP_RETURN_IF_ERROR(
      builder->Add(std::string(kIndexTensorName), std::move(index_t)));
  FCP_ASSIGN_OR_RETURN(auto checkpoint, builder->Build());

  if (input.blob_id.empty()) {
    return absl::InvalidArgumentError("Missing input blob id.");
  }

  if (!context.EmitEncrypted(
          /*reencryption_key_index=*/0,
          Session::KV(std::move(input.key), std::string(std::move(checkpoint)),
                      std::move(input.blob_id)))) {
    return absl::InvalidArgumentError("Emit failed.");
  }
  return absl::OkStatus();
}

}  // anonymous namespace

absl::StatusOr<std::unique_ptr<FnFactory>> ProvideNNHistogramFnFactory(
    const Any& configuration, const Any& config_constraints,
    const WriteConfigurationMap& write_configuration_map,
    ReadRecordFn read_record_fn, NearestNeighborFn nn_fn) {
  NNHistogramContainerInitializeConfiguration init_config;
  if (!configuration.UnpackTo(&init_config)) {
    return absl::InvalidArgumentError(
        "Cannot unpack init config to "
        "NNHistogramContainerInitializeConfiguration.");
  }
  if (!write_configuration_map.contains(
          init_config.synthetic_data_embeddings_configuration_id())) {
    return absl::InvalidArgumentError(
        "Write configuration map doesn't contain synthetic data embeddings "
        "configuration id.");
  }
  std::string path = write_configuration_map.at(
      init_config.synthetic_data_embeddings_configuration_id());
  FCP_ASSIGN_OR_RETURN(std::vector<Embedding> embeddings, read_record_fn(path));
  return std::make_unique<NNHistogramFnFactory>(std::move(embeddings),
                                                std::move(nn_fn));
}
}  // namespace confidential_federated_compute::nn_histogram
