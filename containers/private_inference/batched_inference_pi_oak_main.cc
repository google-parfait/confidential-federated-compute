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

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "containers/batched_inference/batched_inference_engine.h"
#include "containers/batched_inference/batched_inference_server.h"
#include "pi_batched_inference_provider.h"
#include "pi_client.h"

ABSL_FLAG(std::string, server_address, "", "Address of the Pi Server");
ABSL_FLAG(std::string, feature_name, "FEATURE_NAME_PSI_MEMORY_GENERATION",
          "Feature name for the Pi Server");

const int kIncomingPort = 8080;

namespace confidential_federated_compute::private_inference {
namespace {

using ::confidential_federated_compute::batched_inference::
    BatchedInferenceEngineProvider;
using ::confidential_federated_compute::batched_inference::
    BatchedInferenceServer;
using ::confidential_federated_compute::batched_inference::
    CreateBatchedInferenceServer;
using ::oak::containers::sdk::InstanceEncryptionKeyHandle;
using ::oak::containers::sdk::InstanceSigningKeyHandle;
using ::oak::containers::sdk::OrchestratorClient;

void RunServer() {
  ::private_inference::proto::FeatureName feature_name;
  CHECK(::private_inference::proto::FeatureName_Parse(
      absl::GetFlag(FLAGS_feature_name), &feature_name))
      << "Failed to parse feature name: " << absl::GetFlag(FLAGS_feature_name);

  std::shared_ptr<BatchedInferenceEngineProvider> provider =
      std::make_shared<PiBatchedInferenceProvider>(
          absl::GetFlag(FLAGS_server_address), feature_name);

  absl::StatusOr<std::unique_ptr<BatchedInferenceServer>> server =
      CreateBatchedInferenceServer(
          provider, kIncomingPort, std::make_unique<InstanceSigningKeyHandle>(),
          std::make_unique<InstanceEncryptionKeyHandle>());
  CHECK_OK(server);
  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server.value()->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::private_inference

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  confidential_federated_compute::private_inference::RunServer();
  absl::SleepFor(absl::Seconds(1));
  return 0;
}
