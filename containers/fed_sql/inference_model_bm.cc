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

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "benchmark/benchmark.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/sql/input.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "inference_model.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

ABSL_FLAG(std::string, tokenizer_path, "/saved_model/tokenizer.spm",
          "Path to the tokenizer file.");
ABSL_FLAG(std::string, model_path, "/saved_model/gemma-2-2b-it-sfp.bin",
          "Path to the model weights file.");

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::Input;
using ::fcp::confidentialcompute::BlobHeader;
using ::tensorflow_federated::aggregation::Tensor;

// Synthetic dataset transcripts from gen_checkpoints_main.cc
const std::vector<std::string> kSyntheticTranscripts = {
    "User: Hey Google, can you set a timer for 10 minutes?\n"
    "Assistant: Sure, I set a timer for 10 minutes from now.",

    "User: Hey Google, what's the weather like in London?\n"
    "Assistant: The weather in London today is sunny with a high "
    "of 55 degrees Fahrenheit and a low of 55 degrees Fahrenheit.",

    "User: Hey Google, can you play some music?\n"
    "Assistant: Sure, what kind of music would you like to hear?\n"
    "User: I'm in the mood for something upbeat.\n"
    "Assistant: Okay, playing \"Happy\" by Pharrell Williams on Spotify."};

constexpr char kPromptTemplate[] =
    "Below is the transcript between a user and the user's virtual "
    "assistant.\n"
    "\n"
    "{{transcript}}\n"
    "\n"
    "\n"
    "What is the topic of this transcript? Topic should be less than 3 "
    "words.\n";

class GemmaInferenceBenchmark : public benchmark::Fixture {
 public:
  InferenceModel* model() const { return model_.get(); }

  void SetUp(::benchmark::State& state) override {
    std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
    std::string model_weight_path = absl::GetFlag(FLAGS_model_path);

    int max_generated_tokens = state.range(1);

    SessionInferenceConfiguration inference_configuration;
    inference_configuration.initialize_configuration.mutable_inference_config()
        ->mutable_runtime_config()
        ->set_max_generated_tokens(max_generated_tokens);

    auto* inference_task = inference_configuration.initialize_configuration
                               .mutable_inference_config()
                               ->add_inference_task();
    inference_task->mutable_column_config()->add_input_column_names(
        "transcript");
    inference_task->mutable_column_config()->set_output_column_name("topic");

    // Match the exact prompt template from inference_config_gemma_cpp.textproto
    inference_task->mutable_prompt()->set_prompt_template(kPromptTemplate);

    inference_configuration.initialize_configuration
        .mutable_gemma_init_config()
        ->set_tokenizer_configuration_id("tokenizer_configuration_id");

    inference_configuration.gemma_configuration.emplace();
    inference_configuration.gemma_configuration->tokenizer_path =
        tokenizer_path;
    inference_configuration.gemma_configuration->model_weight_path =
        model_weight_path;

    model_ = std::make_unique<InferenceModel>();
    absl::Time start_time = absl::Now();
    auto status = model_->BuildModel(inference_configuration);
    if (!status.ok()) {
      state.SkipWithError(
          (std::string("Failed to build model: ") + status.ToString()).c_str());
      return;
    }
    LOG(INFO) << "Initializing model took " << absl::Now() - start_time;
  }

  void TearDown(::benchmark::State& state) override { model_.reset(); }

 private:
  std::unique_ptr<InferenceModel> model_;
};

BENCHMARK_DEFINE_F(GemmaInferenceBenchmark,
                   BM_GemmaInference)(benchmark::State& state) {
  int batch_size = state.range(0);
  std::vector<std::string> prompt_inputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    prompt_inputs[i] = kSyntheticTranscripts[i % kSyntheticTranscripts.size()];
  }
  std::vector<Tensor> columns;
  columns.push_back(Tensor(std::move(prompt_inputs), "transcript"));
  BlobHeader blob_header;
  auto input_or = Input::CreateFromTensors(std::move(columns), blob_header);
  if (!input_or.ok()) {
    state.SkipWithError("Failed to create input tensors.");
    return;
  }

  for (auto _ : state) {
    auto inference_status = model()->RunInference(*input_or);
    if (!inference_status.ok()) {
      state.SkipWithError(
          (std::string("Inference failed: ") + inference_status.ToString())
              .c_str());
      break;
    }
  }

  state.SetItemsProcessed(batch_size * state.iterations());
}

// The first parameter is the batch size.
// The second parameter is `max_generated_tokens`.
BENCHMARK_REGISTER_F(GemmaInferenceBenchmark, BM_GemmaInference)
    ->Args({1, 32})
    ->Args({1, 128})
    ->Args({4, 32})
    ->Args({4, 128})
    ->Args({16, 32})
    ->Args({16, 128});

}  // namespace
}  // namespace confidential_federated_compute::fed_sql

int main(int argc, char** argv) {
  std::vector<char*> remaining_args = absl::ParseCommandLine(argc, argv);
  int remaining_argc = remaining_args.size();
  char** remaining_argv = remaining_args.data();

  ::benchmark::Initialize(&remaining_argc, remaining_argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
