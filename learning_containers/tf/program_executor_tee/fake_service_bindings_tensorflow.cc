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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/program_context/cc/fake_server.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return tensorflow_federated::CreateTensorFlowExecutor();
}

PYBIND11_MODULE(fake_service_bindings_tensorflow, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() =
      "Python bindings for starting fake services, primarily for testing.";

  py::class_<FakeDataReadWriteService>(m, "FakeDataReadWriteService")
      .def(py::init<>())
      .def("store_plaintext_message",
           [](FakeDataReadWriteService& self, absl::string_view uri,
              absl::string_view message) {
             auto result = self.StorePlaintextMessage(uri, message);
             // TODO: It currently appears that using a TF>2.14 pip dependency
             // currently prevents us from handling StatusOr via pybind. Try to
             // avoid throwing a runtime error here once we no longer require
             // the TF pip dependency.
             if (!result.ok()) {
               throw std::runtime_error("Failed to store plaintext message: " +
                                        std::string(result.message()));
             }
           })
      .def("get_read_request_uris",
           &FakeDataReadWriteService::GetReadRequestUris)
      .def("get_write_call_args", &FakeDataReadWriteService::GetWriteCallArgs);

  py::class_<FakeComputationDelegationService>(
      m, "FakeComputationDelegationService")
      .def(py::init([](std::vector<std::string> worker_bns) {
             return new FakeComputationDelegationService(worker_bns,
                                                         CreateExecutor);
           }),
           py::arg("worker_bns"));

  py::class_<FakeServer>(m, "FakeServer")
      .def(py::init<int, FakeDataReadWriteService*,
                    FakeComputationDelegationService*>(),
           py::arg("port"), py::arg("data_read_write_service"),
           py::arg("computation_delegation_service"))
      .def("start", &FakeServer::Start, "Starts the gRPC server.")
      .def("stop", &FakeServer::Stop, "Stops the gRPC server.");
}

}  // namespace confidential_federated_compute::program_executor_tee