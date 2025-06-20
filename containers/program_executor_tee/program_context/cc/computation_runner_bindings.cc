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
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "containers/program_executor_tee/program_context/cc/computation_runner.h"
#include "containers/program_executor_tee/program_context/cc/noise_client_session.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace confidential_federated_compute::program_executor_tee {

struct GrpcInitializer {
  GrpcInitializer() {
    std::cout << "Initializing gRPC in computation runner bindings..."
              << std::endl;
    grpc_init();
  }
  ~GrpcInitializer() {
    std::cout << "Shutting down gRPC in computation runner bindings..."
              << std::endl;
    grpc_shutdown();
  }
};

PYBIND11_MODULE(computation_runner_bindings, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  static GrpcInitializer grpc_initializer;

  pybind11::class_<ComputationDelegationResult>(m,
                                                "ComputationDelegationResult")
      .def(pybind11::init<>())
      .def_readwrite("response", &ComputationDelegationResult::response)
      .def_property(
          "status",
          [](ComputationDelegationResult& self) { return self.status; },
          [](ComputationDelegationResult& self, const grpc::Status& status) {
            self.status = status;
          });

  pybind11::class_<ComputationRunner>(m, "ComputationRunner")
      .def(pybind11::init<
               std::vector<std::string>,
               std::optional<std::function<ComputationDelegationResult(
                   ::fcp::confidentialcompute::outgoing::ComputationRequest)>>,
               std::string>(),
           pybind11::arg("worker_bns"),
           pybind11::arg("comp_delegation_proxy") = std::nullopt,
           pybind11::arg("attester_id") = "")
      .def("invoke_comp",
           [](ComputationRunner& self, int num_clients,
              tensorflow_federated::v0::Value comp,
              std::optional<tensorflow_federated::v0::Value> arg) {
             auto result =
                 self.InvokeComp(num_clients, std::move(comp), std::move(arg));
             // TODO: It currently appears that using a TF>2.14 pip dependency
             // currently prevents us from handling StatusOr via pybind. Try to
             // avoid throwing a runtime error here once we no longer require
             // the TF pip dependency.
             if (!result.ok()) {
               throw std::runtime_error("Failed to execute computation: " +
                                        std::string(result.status().message()));
             } else {
               return *result;
             }
           });
}

}  // namespace confidential_federated_compute::program_executor_tee
