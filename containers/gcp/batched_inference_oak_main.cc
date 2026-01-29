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

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "attestation_policy.pb.h"
#include "batched_inference.pb.h"
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "client.h"
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/batched_inference/batched_inference_server.h"
#include "google/protobuf/text_format.h"
#include "grpcpp/grpcpp.h"
#include "proto_parsing_utils.h"

ABSL_FLAG(std::string, server_address, "localhost", "Server address.");

ABSL_FLAG(std::string, policy_path, "",
          "Path to the AttestationPolicy textproto file.");

ABSL_FLAG(std::string, jwks_path, "",
          "Path to local JWKS file for offline verification.");

ABSL_FLAG(bool, dump_jwt, false, "Dump JWT payload (DEBUG).");

namespace confidential_federated_compute::gcp {
namespace {

const int SERVER_PORT = 8000;
const int CLIENT_PORT = 8080;

using ::oak::containers::sdk::InstanceEncryptionKeyHandle;
using ::oak::containers::sdk::InstanceSigningKeyHandle;
using ::oak::containers::sdk::OrchestratorClient;
using ::oak::crypto::EncryptionKeyHandle;
using ::oak::crypto::SigningKeyHandle;

using batched_inference::BatchedInferenceProvider;
using batched_inference::BatchedInferenceServer;

AttestationPolicy ReadPolicyOrDie() {
  std::string policy_path = absl::GetFlag(FLAGS_policy_path);
  if (policy_path.empty()) {
    LOG(FATAL) << "--policy_path must be specified.";
  }
  AttestationPolicy policy;
  ReadTextProtoOrDie(policy_path, &policy);
  return policy;
}

std::string ReadJwksPayloadOrDie() {
  std::string jwks_path = absl::GetFlag(FLAGS_jwks_path);
  CHECK(!jwks_path.empty()) << "JWKS path is empty.";
  LOG(INFO) << "Loading JWKS from local file: " << jwks_path;
  std::ifstream jwks_file(jwks_path);
  CHECK(jwks_file.is_open()) << "Failed to open local JWKS file.";
  std::stringstream buffer;
  buffer << jwks_file.rdbuf();
  std::string jwks_payload_data = buffer.str();
  return jwks_payload_data;
}

absl::StatusOr<std::vector<absl::StatusOr<std::string>>>
IssueBatchedInferenceRequest(Client* client, std::vector<std::string> prompts) {
  BatchedInferenceRequest batch_request;
  for (const auto& prompt : prompts) {
    auto* req_item = batch_request.add_requests();
    req_item->set_text(prompt);
  }
  batch_request.mutable_params()->set_max_output_tokens(1024);

  std::string serialized_request;
  if (!batch_request.SerializeToString(&serialized_request)) {
    return absl::InternalError("Failed to serialize request proto");
  }

  absl::StatusOr<std::string> response_or = client->Invoke(serialized_request);
  if (!response_or.ok()) {
    return response_or.status();
  }

  BatchedInferenceResponse batch_response;
  if (!batch_response.ParseFromString(*response_or)) {
    return absl::InternalError("Failed to parse GCP response");
  }

  if (batch_response.results_size() != prompts.size()) {
    return absl::InternalError(absl::StrCat("Size mismatch: sent ",
                                            prompts.size(), ", got ",
                                            batch_response.results_size()));
  }

  std::vector<absl::StatusOr<std::string>> outputs;
  for (const auto& result : batch_response.results()) {
    if (result.status().code() == 0) {  // OK
      outputs.push_back(result.text());
    } else {
      outputs.push_back(
          absl::InternalError(absl::StrCat(result.status().message())));
    }
  }
  return outputs;
}

class TestBatchedInferenceProviderImpl : public BatchedInferenceProvider {
 public:
  explicit TestBatchedInferenceProviderImpl(std::unique_ptr<Client> client)
      : client_(std::move(client)) {}

  virtual ~TestBatchedInferenceProviderImpl() {}

  std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) override {
    absl::StatusOr<std::vector<absl::StatusOr<std::string>>> replies_or =
        IssueBatchedInferenceRequest(client_.get(), prompts);
    if (replies_or.ok()) {
      return *replies_or;
    } else {
      absl::StatusOr<std::string> error = replies_or.status();
      return std::vector<absl::StatusOr<std::string>>(prompts.size(), error);
    }
  }

 private:
  std::unique_ptr<Client> client_;
};

void RunServer() {
  std::string server_target = absl::StrFormat(
      "%s:%d", absl::GetFlag(FLAGS_server_address), SERVER_PORT);

  LOG(INFO) << "Connecting to server at " << server_target;

  AttestationPolicy attestation_policy = ReadPolicyOrDie();
  std::string jwks_payload = ReadJwksPayloadOrDie();
  bool dump_jwt = absl::GetFlag(FLAGS_dump_jwt);

  absl::StatusOr<std::unique_ptr<AttestationTokenVerifier>> verifier_or =
      CreateAttestationTokenVerifier(attestation_policy, jwks_payload,
                                     dump_jwt);
  CHECK_OK(verifier_or.status())
      << "Couldn't create an attestation verifier: " << verifier_or.status();

  absl::StatusOr<std::unique_ptr<Client>> client_or =
      CreateClient(server_target, std::move(*verifier_or));
  CHECK_OK(client_or.status())
      << "Failed to create a GCP client: " << client_or.status();

  absl::StatusOr<std::unique_ptr<BatchedInferenceServer>> server_or =
      CreateBatchedInferenceServer(
          std::make_shared<TestBatchedInferenceProviderImpl>(
              std::move(*client_or)),
          CLIENT_PORT, std::make_unique<InstanceSigningKeyHandle>(),
          std::make_unique<InstanceEncryptionKeyHandle>());

  CHECK_OK(server_or.status())
      << "Couldn't create the batched inference server: " << server_or.status();

  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server_or.value()->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::gcp

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  confidential_federated_compute::gcp::RunServer();
  absl::SleepFor(absl::Seconds(1));
  return 0;
}
