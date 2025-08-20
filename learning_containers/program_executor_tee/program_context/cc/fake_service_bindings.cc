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

#include <grpcpp/grpcpp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "grpcpp/server_context.h"
#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

class FakeServer {
 public:
  FakeServer(int port, FakeDataReadWriteService* data_read_write_service,
             FakeComputationDelegationService* computation_delegation_service)
      : server_address_("[::1]:" + std::to_string(port)),
        data_read_write_service_(data_read_write_service),
        computation_delegation_service_(computation_delegation_service) {}

  void Start() {
    if (server_) {
      throw std::runtime_error("Server is already running.");
    }
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    if (data_read_write_service_) {
      builder.RegisterService(data_read_write_service_);
    }
    if (computation_delegation_service_) {
      builder.RegisterService(computation_delegation_service_);
    }
    server_ = builder.BuildAndStart();
    if (!server_) {
      throw std::runtime_error("Could not start server on " + server_address_);
    }
  }

  void Stop() {
    if (!server_) {
      return;  // It's often better to make Stop idempotent.
    }
    server_->Shutdown();
  }

  // The destructor will ensure Stop() is called.
  ~FakeServer() {
    if (server_) {
      Stop();
    }
  }

 private:
  std::string server_address_;
  FakeDataReadWriteService* data_read_write_service_;
  FakeComputationDelegationService* computation_delegation_service_;
  std::unique_ptr<grpc::Server> server_;
};

PYBIND11_MODULE(fake_service_bindings, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() =
      "Python bindings for starting fake services, primarily for testing.";

  py::class_<FakeDataReadWriteService>(m, "FakeDataReadWriteService")
      .def(pybind11::init<>())
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
      .def(pybind11::init<std::vector<std::string>>());

  py::class_<FakeServer>(m, "FakeServer")
      .def(py::init<int, FakeDataReadWriteService*,
                    FakeComputationDelegationService*>(),
           py::arg("port"), py::arg("data_read_write_service"),
           py::arg("computation_delegation_service"))
      .def("start", &FakeServer::Start, "Starts the gRPC server.")
      .def("stop", &FakeServer::Stop, "Stops the gRPC server.");
}

}  // namespace confidential_federated_compute::program_executor_tee