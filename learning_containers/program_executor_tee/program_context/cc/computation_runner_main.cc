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
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "learning_containers/program_executor_tee/program_context/cc/computation_runner.h"

ABSL_FLAG(int32_t, computatation_runner_port, 10000,
          "Port to run the computation runner on.");
ABSL_FLAG(std::string, outgoing_server_address, "",
          "The address at which the untrusted root server can be reached for "
          "data read/write requests and computation delegation requests.");
ABSL_FLAG(std::vector<std::string>, worker_bns, {},
          "A list of worker bns addresses.");
ABSL_FLAG(std::string, attester_id, "",
          "The attester id for setting up the noise session.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const std::string server_address =
      absl::StrCat("[::1]:", absl::GetFlag(FLAGS_computatation_runner_port));
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  auto computation_runner_service = std::make_unique<
      confidential_federated_compute::program_executor_tee::ComputationRunner>(
      absl::GetFlag(FLAGS_worker_bns), absl::GetFlag(FLAGS_attester_id),
      absl::GetFlag(FLAGS_outgoing_server_address));

  builder.RegisterService(computation_runner_service.get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server == nullptr) {
    LOG(ERROR) << "Computation runner failed to start. Check the logs above "
                  "for information.";
    return -1;
  }
  LOG(INFO) << "Computation runner started, listening on " << server_address;
  server->Wait();
}