// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "include/llama.h"

namespace confidential_federated_compute::test_llama_cpp {
namespace {

absl::Status GenerateOutput(std::string *output) {
  // Number of layers to offload to the GPU.
  int ngl = 999;
  // Number of tokens to predict.
  int n_predict = 1024;
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = ngl;
  // This is a local path to the model weights file within the container.
  // If the other model is used, the file name needs to be updated.
  std::string model_path = "/saved_model/gemma-3-12b-it-q4_0.gguf";
  llama_model *model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  const llama_vocab *vocab = llama_model_get_vocab(model);
  if (model == NULL) {
    return absl::InternalError("Unable to load model");
  }
  std::string prompt = "Tell three facts about cats.";
  // Tokenize the prompt &  find the number of tokens in the prompt
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       NULL, 0, true, true);
  // Allocate space for the tokens and tokenize the prompt.
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                     prompt_tokens.size(), true, true) < 0) {
    return absl::InternalError("Failed to tokenize the prompt");
  }
  // Initialize the context.
  llama_context_params ctx_params = llama_context_default_params();
  // n_ctx is the context size.
  ctx_params.n_ctx = n_prompt + n_predict - 1;
  // n_batch is the maximum number of tokens that can be processed in a single
  // call to llama_decode.
  ctx_params.n_batch = n_prompt;
  // Enable performance counters.
  ctx_params.no_perf = false;
  llama_context *ctx = llama_init_from_model(model, ctx_params);
  if (ctx == NULL) {
    return absl::InternalError("Failed to create the llama_context");
  }
  // Initialize the sampler.
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  llama_sampler *smpl = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  // Add the prompt to the output.
  for (auto id : prompt_tokens) {
    char buf[128];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
      return absl::InternalError("Failed to convert token to piece");
    }
    std::string s(buf, n);
    absl::StrAppend(output, s);
  }
  // Prepare a batch for the prompt.
  llama_batch batch =
      llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
  // Main loop.
  const auto t_main_start = ggml_time_us();
  int n_decode = 0;
  llama_token new_token_id;
  for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
    // Evaluate the current batch.
    if (llama_decode(ctx, batch)) {
      return absl::InternalError("Failed to eval");
    }
    n_pos += batch.n_tokens;
    // Sample the next token.
    {
      new_token_id = llama_sampler_sample(smpl, ctx, -1);
      // Is it an end of generation?
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
      }
      char buf[128];
      int n =
          llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
      if (n < 0) {
        return absl::InternalError("Failed to convert token to piece");
      }
      std::string s(buf, n);
      absl::StrAppend(output, s);
      // Prepare the next batch with the sampled token.
      batch = llama_batch_get_one(&new_token_id, 1);
      n_decode += 1;
    }
  }
  const auto t_main_end = ggml_time_us();
  absl::StrAppend(
      output,
      absl::StrCat("\nSpeed: ",
                   std::to_string(n_decode /
                                  ((t_main_end - t_main_start) / 1000000.0f))));
  fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
          __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
          n_decode / ((t_main_end - t_main_start) / 1000000.0f));
  llama_perf_sampler_print(smpl);
  llama_perf_context_print(ctx);
  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);
  return absl::OkStatus();
}

}  // namespace
}  // namespace confidential_federated_compute::test_llama_cpp

int main(int argc, char **argv) {
  std::string output;
  absl::Status output_status =
      confidential_federated_compute::test_llama_cpp::GenerateOutput(&output);
  if (output_status.ok()) {
    LOG(INFO) << &output << "Success.\n";
  } else {
    LOG(INFO) << &output << "Failure:\n" << output_status.ToString();
  }
  return 0;
}
