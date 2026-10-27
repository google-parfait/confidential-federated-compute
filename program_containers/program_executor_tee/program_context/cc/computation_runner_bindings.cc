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

#include "program_executor_tee/program_context/cc/computation_runner_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "federated_language/proto/computation.pb.h"
#include "program_executor_tee/program_context/cc/computation_runner.h"
#include "program_executor_tee/program_context/cc/value_conversion.h"
#include "proto/attestation/reference_value.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

using ::tensorflow_federated::v0::Value;

void DefineComputationRunner(py::module_& m,
                             LeafExecutorFactory create_leaf_executor) {
  py::class_<ComputationRunner>(m, "ComputationRunner")
      .def(
          py::init([create_leaf_executor = std::move(create_leaf_executor)](
                       std::vector<std::string> worker_bns,
                       std::string serialized_reference_values,
                       std::string outgoing_server_address,
                       bool use_elastic_composing_executor,
                       bool use_mergeable_execution_context,
                       int max_concurrent_computation_calls) {
            // Decode and parse the serialized reference values. This is needed
            // because the reference values are base64 encoded in the Python
            // execution context.
            std::string binary_data;
            if (!absl::Base64Unescape(serialized_reference_values,
                                      &binary_data)) {
              throw py::value_error(
                  "Failed to unescape serialized reference values.");
            }
            oak::attestation::v1::ReferenceValues reference_values;
            if (!reference_values.ParseFromString(binary_data)) {
              throw py::value_error(
                  "Failed to parse serialized reference values.");
            }
            return std::make_unique<ComputationRunner>(
                [create_leaf_executor, max_concurrent_computation_calls]() {
                  return create_leaf_executor(max_concurrent_computation_calls);
                },
                worker_bns, reference_values.SerializeAsString(),
                outgoing_server_address, use_elastic_composing_executor,
                use_mergeable_execution_context);
          }),
          py::arg("worker_bns") = std::vector<std::string>(),
          py::arg("serialized_reference_values") = "",
          py::arg("outgoing_server_address") = "",
          py::arg("use_elastic_composing_executor") = false,
          py::arg("use_mergeable_execution_context") = false,
          py::arg("max_concurrent_computation_calls") = -1)
      // Executes the computation in `fn`, a serialized `executor_pb2.Value`,
      // and returns the leaves of the result (see `EncodeValue`).
      // `arg_leaves` holds the leaves of the argument (see `DecodeValue`),
      // or is None if there is no argument.
      .def(
          "execute",
          [](ComputationRunner& self, py::handle fn, py::handle arg_leaves,
             int num_clients,
             const std::string& target_worker_bns) -> py::object {
            Value fn_pb;
            py::bytes serialized_fn = fn.cast<py::bytes>();
            if (!fn_pb.ParseFromString(absl::string_view(serialized_fn))) {
              throw py::value_error(
                  "Failed to parse serialized tensorflow_federated.v0.Value.");
            }
            const federated_language::FunctionType& fn_type =
                fn_pb.computation().type().function();
            std::optional<Value> arg;
            if (!arg_leaves.is_none()) {
              DecodeValue(fn_type.parameter(), arg_leaves, &arg.emplace());
            }
            absl::StatusOr<Value> result;
            {
              py::gil_scoped_release release;
              result = self.Execute(fn_pb, std::move(arg), num_clients,
                                    target_worker_bns);
            }
            if (!result.ok()) {
              throw std::runtime_error(std::string(result.status().message()));
            }
            return EncodeValue(fn_type.result(), *result);
          },
          py::arg("fn"), py::arg("arg_leaves"), py::arg("num_clients") = 0,
          py::arg("target_worker_bns") = "");
}

}  // namespace confidential_federated_compute::program_executor_tee
