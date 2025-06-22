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
#include "containers/program_executor_tee/confidential_transform_server.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <execution>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute::program_executor_tee {
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Struct;

absl::Status ProgramExecutorTeeSession::ConfigureSession(
    fcp::confidentialcompute::SessionRequest configure_request) {
  return absl::OkStatus();
}

absl::StatusOr<SessionResponse> ProgramExecutorTeeSession::SessionWrite(
    const WriteRequest& write_request, std::string unencrypted_data) {
  return absl::UnimplementedError(
      "SessionWrite is not supported in program executor TEE.");
}

absl::StatusOr<SessionResponse> ProgramExecutorTeeSession::FinalizeSession(
    const FinalizeRequest& request, const BlobMetadata& input_metadata) {
  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();

  std::vector<std::string> worker_bns_addresses;
  worker_bns_addresses.reserve(
      initialize_config_.worker_bns_addresses().size());
  for (const auto& address : initialize_config_.worker_bns_addresses()) {
    worker_bns_addresses.push_back(address);
  }

  // TODO: Allow the attester_id to be configured.
  std::string attester_id = "fake_attester";

  pybind11::scoped_interpreter guard{};
  try {
    // Load the python function for running the program.
    auto run_program =
        pybind11::module::import(
            "containers.program_executor_tee.program_context.program_runner")
            .attr("run_program");

    // Schedule execution of the program as a Task.
    pybind11::object task = pybind11::module::import("asyncio").attr(
        "ensure_future")(run_program(initialize_config_.program(),
                                     initialize_config_.outgoing_server_port(),
                                     worker_bns_addresses, attester_id));

    // Run the task in the event loop and get the result.
    pybind11::object loop =
        pybind11::module::import("asyncio").attr("get_event_loop")();
    pybind11::object result = loop.attr("run_until_complete")(task);

    read_response->set_finish_read(true);
  } catch (const std::exception& e) {
    LOG(INFO) << "Error executing federated program: " << e.what();
    read_response->set_finish_read(false);
  }

  return response;
}

absl::StatusOr<google::protobuf::Struct>
ProgramExecutorTeeConfidentialTransform::StreamInitializeTransform(
    const fcp::confidentialcompute::InitializeRequest* request) {
  ProgramExecutorTeeInitializeConfig config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "ProgramExecutorTeeInitializeConfig cannot be unpacked.");
  }
  initialize_config_ = std::move(config);

  Struct config_properties;
  (*config_properties.mutable_fields())["program"].set_string_value(
      initialize_config_.program());
  return config_properties;
}

}  // namespace confidential_federated_compute::program_executor_tee
