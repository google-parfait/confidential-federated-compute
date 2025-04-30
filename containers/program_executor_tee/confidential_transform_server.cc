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
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute::program_executor_tee {
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteRequest;

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
  // TODO: Run the program in the config provided at initialization time,
  // instead of using a hardcoded program.
  auto program = R"(
import federated_language
import numpy as np

async def trusted_program(release_manager):

  client_data_type = federated_language.FederatedType(
      np.int32, federated_language.CLIENTS
  )

  @federated_language.federated_computation(client_data_type)
  def my_comp(client_data):
    return federated_language.federated_sum(client_data)

  result = await my_comp([1, 2])

  await release_manager.release(result, "result1")
  )";

  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();

  pybind11::scoped_interpreter guard{};
  try {
    // Load the python function for running the program.
    auto run_program =
        pybind11::module::import(
            "containers.program_executor_tee.program_context.program_runner")
            .attr("run_program");

    // Schedule execution of the program as a Task.
    pybind11::object task = pybind11::module::import("asyncio").attr(
        "ensure_future")(run_program(program, kDataReadWriteServicePort));

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
}  // namespace confidential_federated_compute::program_executor_tee
