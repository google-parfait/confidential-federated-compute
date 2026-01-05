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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/batched_inference/batched_inference_server.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using oak::containers::sdk::InstanceEncryptionKeyHandle;
using oak::containers::sdk::InstanceSigningKeyHandle;
using ::oak::containers::sdk::OrchestratorClient;
using ::oak::crypto::EncryptionKeyHandle;
using ::oak::crypto::SigningKeyHandle;

const int INCOMING_PORT = 8080;

class TestBatchedInferenceProviderImpl : public BatchedInferenceProvider {
 public:
  virtual ~TestBatchedInferenceProviderImpl() {}

  std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) override {
    std::vector<absl::StatusOr<std::string>> results;
    for (const auto& prompt : prompts) {
      results.push_back("Processed: " + prompt);
    }
    return results;
  }
};

void RunServer() {
  absl::StatusOr<std::unique_ptr<BatchedInferenceServer>> server =
      CreateBatchedInferenceServer(
          std::make_shared<TestBatchedInferenceProviderImpl>(), INCOMING_PORT,
          std::make_unique<InstanceSigningKeyHandle>(),
          std::make_unique<InstanceEncryptionKeyHandle>());
  CHECK_OK(server);
  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server.value()->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::batched_inference

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  confidential_federated_compute::batched_inference::RunServer();
  absl::SleepFor(absl::Seconds(1));
  return 0;
}
