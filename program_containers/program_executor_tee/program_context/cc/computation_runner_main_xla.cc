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
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "program_executor_tee/program_context/cc/computation_runner.h"
#include "proto/attestation/reference_value.pb.h"

ABSL_FLAG(int32_t, computatation_runner_port, 10000,
          "Port to run the computation runner on.");
ABSL_FLAG(std::string, outgoing_server_address, "",
          "The address at which the untrusted root server can be reached for "
          "data read/write requests and computation delegation requests.");
ABSL_FLAG(std::vector<std::string>, worker_bns, {},
          "A list of worker bns addresses.");
ABSL_FLAG(std::string, serialized_reference_values, "",
          "The serialized reference values of the program worker for setting "
          "up the client noise session.");

// The default gRPC message size is 4 KiB. Increase it to 100 KiB.
constexpr int kMaxGrpcMessageSize = 100 * 1024 * 1024;

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return federated_language_jax::CreateXLAExecutor();
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const std::string server_address =
      absl::StrCat("[::1]:", absl::GetFlag(FLAGS_computatation_runner_port));
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kMaxGrpcMessageSize);
  builder.SetMaxSendMessageSize(kMaxGrpcMessageSize);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Decode and parse the serialized reference values. This is needed because
  // the reference values are base64 encoded in the Python execution context.
  std::string binary_data;
  if (!absl::Base64Unescape(absl::GetFlag(FLAGS_serialized_reference_values),
                            &binary_data)) {
    LOG(ERROR) << "Failed to unescape serialized reference values.";
    return -1;
  }
  oak::attestation::v1::ReferenceValues reference_values;
  if (!reference_values.ParseFromString(binary_data)) {
    LOG(ERROR) << "Failed to parse serialized reference values.";
    return -1;
  }

  auto computation_runner_service = std::make_unique<
      confidential_federated_compute::program_executor_tee::ComputationRunner>(
      CreateExecutor, absl::GetFlag(FLAGS_worker_bns),
      reference_values.SerializeAsString(),
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