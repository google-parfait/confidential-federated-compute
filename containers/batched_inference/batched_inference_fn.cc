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

#include "containers/batched_inference/batched_inference_fn.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/fns/do_fn.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using ::confidential_federated_compute::Session;

// A class that batches all writes received until a Commit(), and then
// issues a single call to the inference provider.
// FUTURE WORK(b/452094015): Consider adding a max batch size argument
// to enable forking out smaller batches than what is implied by the
// presence of Commit() in the write stream, if needed.
class BatchedInferenceFn final
    : public confidential_federated_compute::fns::DoFn {
 public:
  explicit BatchedInferenceFn(
      std::shared_ptr<BatchedInferenceProvider> batched_inference_provider)
      : batched_inference_provider_(batched_inference_provider) {}

  ~BatchedInferenceFn() {}

  absl::Status Do(Session::KV input, Context& context) override {
    input_batch_.push_back(std::move(input));
    return absl::OkStatus();
  }

  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override {
    std::vector<Session::KV> current_batch;
    std::swap(current_batch, input_batch_);
    std::vector<std::string> prompts;
    for (Session::KV& input : current_batch) {
      prompts.push_back(std::move(input.data));
    }
    std::vector<absl::StatusOr<std::string>> results =
        batched_inference_provider_->DoBatchedInference(prompts);
    if (results.size() != prompts.size()) {
      return absl::InternalError(absl::StrCat(
          "The number of results (", results.size(),
          ") does not match the number of prompts (", prompts.size(), ")."));
    }
    for (int i = 0; i < current_batch.size(); ++i) {
      // For now, the failure of inference of encryption on any element in the
      // batch will fail the entire batch. To discuss and reviset as needed.
      // FUTURE WORK(b/452094015): Consider making this behavior more
      // configurable if needed.
      const Session::KV& input = current_batch[i];
      absl::StatusOr<std::string>& output = results[i];
      if (!output.ok()) {
        return absl::InternalError(
            absl::StrCat("Inference failed: ", output.status().ToString()));
      }
      // NOTE: We are intentionally using the same blob ids in the output as
      // those used in the corresponding inputs.
      // FUTURE WORK(b/452094015): Consider making this behavior configurable
      // if needed.
      if (!context.EmitEncrypted(
              0, Session::KV{input.key, *std::move(output), input.blob_id})) {
        return absl::InternalError(absl::StrCat("EmitEncrypted failed"));
      }
    }
    return fcp::confidentialcompute::CommitResponse();
  }

 private:
  std::shared_ptr<BatchedInferenceProvider> batched_inference_provider_;
  std::vector<Session::KV> input_batch_;
};

class BatchedInferenceFnFactory
    : public confidential_federated_compute::fns::FnFactory {
 public:
  explicit BatchedInferenceFnFactory(
      std::shared_ptr<BatchedInferenceProvider> batched_inference_provider)
      : batched_inference_provider_(batched_inference_provider) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<BatchedInferenceFn>(batched_inference_provider_);
  }

 private:
  std::shared_ptr<BatchedInferenceProvider> batched_inference_provider_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<fns::FnFactory>> CreateBatchedInferenceFnFactory(
    std::shared_ptr<BatchedInferenceProvider> batched_inference_provider) {
  return std::make_unique<BatchedInferenceFnFactory>(
      batched_inference_provider);
}

}  // namespace confidential_federated_compute::batched_inference