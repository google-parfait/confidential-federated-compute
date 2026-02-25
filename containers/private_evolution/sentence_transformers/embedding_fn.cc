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
#include "embedding_fn.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "archive_utils.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.pb.h"
#include "tensor_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {

using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::
    SentenceTransformersContainerInitializeConfiguration;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_STRING;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;

absl::StatusOr<std::string> ExtractBlobId(const BlobMetadata& blob_metadata) {
  if (!blob_metadata.has_hpke_plus_aead_data()) {
    return absl::InvalidArgumentError(
        "Blob meta data doesn't contain HPKE plus AEAD data.");
  }
  if (!blob_metadata.hpke_plus_aead_data().blob_id().empty()) {
    return blob_metadata.hpke_plus_aead_data().blob_id();
  }
  auto associated_data =
      blob_metadata.hpke_plus_aead_data().ciphertext_associated_data();
  BlobHeader header;
  if (!header.ParseFromString(associated_data)) {
    return absl::InvalidArgumentError(
        "Cipher text associated data is not a valid BlobHeader.");
  }
  return header.blob_id();
}

// Implementation of Fn interface that generates embeddings for input data.
// Not thread-safe.
class EmbeddingFn final : public Fn {
 public:
  EmbeddingFn(absl::string_view model_artifact_path,
              std::optional<std::string> prompt,
              std::unique_ptr<ModelDelegate> model_delegate);

  // Initialize embedding model, and initialize python runtime when needed.
  absl::Status InitializeReplica(google::protobuf::Any config,
                                 Context& context) override;

  // Finalize python runtime when needed.
  absl::Status FinalizeReplica(Any config, Context& context) override;

  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // No-op.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override final {
    return fcp::confidentialcompute::CommitResponse();
  }

 private:
  // Generate embeddings for a given batch of input sequences.
  // Input: a FederatedComputeCheckpoint with a string tensor, the name of the
  // tensor is kDataTensorName, the values inside the tensor are utf-8 encoded
  // string. Output: a FederatedComputeCheckpoint with a 2D float tensor, the
  // name of the tensor is kDataTensorName. First dimension of the tensor is the
  // batch dimension. The second dimension is the embedding dimension.
  absl::StatusOr<std::string> Process(std::string unencrypted_data);

  std::string model_artifact_path_;
  std::optional<std::string> prompt_;
  std::unique_ptr<ModelDelegate> delegate_;
};

EmbeddingFn::EmbeddingFn(absl::string_view model_artifact_path,
                         std::optional<std::string> prompt,
                         std::unique_ptr<ModelDelegate> model_delegate) {
  model_artifact_path_ = std::string(model_artifact_path);
  prompt_ = prompt;
  delegate_ = std::move(model_delegate);
};

absl::Status EmbeddingFn::InitializeReplica(Any config, Context& context) {
  if (!delegate_->InitializeModel(model_artifact_path_)) {
    return absl::InvalidArgumentError("Model initialization failed");
  }
  LOG(WARNING) << "Replica initialized.";
  return absl::OkStatus();
}

absl::Status EmbeddingFn::FinalizeReplica(Any config, Context& context) {
  LOG(WARNING) << "Replica finalized.";
  return absl::OkStatus();
}

absl::StatusOr<WriteFinishedResponse> EmbeddingFn::Write(
    WriteRequest write_request, std::string unencrypted_data,
    Context& context) {
  int64_t unencrypted_data_size = unencrypted_data.size();
  absl::StatusOr<std::string> output = Process(std::move(unencrypted_data));
  if (!output.ok()) {
    return ToWriteFinishedResponse(output.status());
  }

  absl::StatusOr<std::string> blob_id =
      ExtractBlobId(std::move(write_request.first_request_metadata()));
  if (!blob_id.ok()) {
    return ToWriteFinishedResponse(blob_id.status());
  }
  if (!context.EmitEncrypted(
          /*reencryption_key_index=*/0,
          Session::KV(std::move(write_request.first_request_configuration()),
                      std::move(*output), std::move(*blob_id)))) {
    return ToWriteFinishedResponse(absl::InvalidArgumentError("Emit failed."));
  }

  WriteFinishedResponse response;
  response.set_committed_size_bytes(unencrypted_data_size);
  return response;
}

absl::StatusOr<std::string> EmbeddingFn::Process(std::string unencrypted_data) {
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(
      auto parser,
      parser_factory.Create(absl::Cord(std::move(unencrypted_data))));
  FCP_ASSIGN_OR_RETURN(auto tensor,
                       parser->GetTensor(std::string(kDataTensorName)));
  if (tensor.dtype() != DT_STRING) {
    return absl::InvalidArgumentError("Unsupported tensor data type.");
  }
  FCP_ASSIGN_OR_RETURN(
      std::vector<std::vector<float>> embeddings,
      delegate_->GenerateEmbeddings(tensor.ToStringVector(), prompt_));

  FCP_ASSIGN_OR_RETURN(auto t, CreateEmbeddingTensor(std::move(embeddings)));
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  FCP_RETURN_IF_ERROR(builder->Add(std::string(kDataTensorName), std::move(t)));
  FCP_ASSIGN_OR_RETURN(absl::Cord ckpt_cord, builder->Build());

  return std::string(std::move(ckpt_cord));
}

class EmbeddingFnFactory
    : public confidential_federated_compute::fns::FnFactory {
 public:
  explicit EmbeddingFnFactory(absl::string_view model_artifacts_path,
                              std::optional<std::string> prompt,
                              ModelDelegateFactory& delegate_factory)
      : model_artifacts_path_(std::string(model_artifacts_path)),
        prompt_(prompt),
        delegate_factory_(delegate_factory) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<EmbeddingFn>(model_artifacts_path_, prompt_,
                                         delegate_factory_.Create());
  }

 private:
  std::string model_artifacts_path_;
  std::optional<std::string> prompt_;
  ModelDelegateFactory& delegate_factory_;
};

}  // namespace

EmbeddingFnFactoryProvider CreateEmbeddingFnFactoryProvider(
    ModelDelegateFactory& delegate_factory, absl::string_view tmp_dir) {
  return [&delegate_factory, tmp_dir](
             const Any& configuration, const Any& config_constraints,
             const WriteConfigurationMap& write_configuration_map)
             -> absl::StatusOr<std::unique_ptr<FnFactory>> {
    SentenceTransformersContainerInitializeConfiguration config;
    if (!configuration.UnpackTo(&config)) {
      return absl::InvalidArgumentError(
          "Configuration cannot be unpacked into "
          "SentenceTransformerContainerInitializeConfiguration");
    }
    if (!write_configuration_map.contains(
            config.model_artifacts_configuration_id())) {
      return absl::InvalidArgumentError(
          "Model artifacts archive is not in the write configuration map.");
    }
    std::string archive_path =
        write_configuration_map.at(config.model_artifacts_configuration_id());
    FCP_ASSIGN_OR_RETURN(std::string model_artifacts_path,
                         ExtractAll(archive_path, tmp_dir));
    std::optional<std::string> prompt;
    if (config.has_encode_config() &&
        !config.encode_config().prompt().empty()) {
      prompt = config.encode_config().prompt();
    }
    return std::make_unique<EmbeddingFnFactory>(model_artifacts_path, prompt,
                                                delegate_factory);
  };
}

}  // namespace confidential_federated_compute::sentence_transformers
