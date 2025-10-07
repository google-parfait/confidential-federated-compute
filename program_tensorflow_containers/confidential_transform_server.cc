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
#include "confidential_transform_server.h"

#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <optional>

#include "absl/strings/escaping.h"
#include "program_executor_tee/program_context/cc/data_parser.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace confidential_federated_compute::tensorflow::program_executor_tee {

using ::confidential_federated_compute::program_executor_tee::DataParser;

PYBIND11_EMBEDDED_MODULE(data_parser, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  pybind11::class_<BlobDecryptor>(m, "BlobDecryptor");

  pybind11::class_<DataParser>(m, "DataParser")
      .def(pybind11::init<BlobDecryptor*>())
      .def("parse_read_response_to_value",
           [](DataParser& self,
              const fcp::confidentialcompute::outgoing::ReadResponse&
                  read_response,
              const std::string& nonce, const std::string& key) {
             auto result =
                 self.ParseReadResponseToValue(read_response, nonce, key);
             // TODO: It currently appears that using a TF>2.14 pip dependency
             // currently prevents us from handling StatusOr via pybind. Try to
             // avoid throwing a runtime error here once we no longer require
             // the TF pip dependency.
             if (!result.ok()) {
               throw std::runtime_error("Failed to parse ReadResponse: " +
                                        std::string(result.status().message()));
             }
             return *result;
           });
}

std::optional<pybind11::function>
TensorflowProgramExecutorTeeConfidentialTransform::GetProgramInitializeFn() {
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config = GetInitializeConfig();

  pybind11::object data_parser_instance =
      pybind11::module::import("data_parser")
          .attr("DataParser")(*GetBlobDecryptor());

  return pybind11::module::import("initialize_program_tensorflow")
      .attr("get_program_initialize_fn")(
          initialize_config.outgoing_server_address(), GetWorkerBnsAddresses(),
          pybind11::bytes(absl::Base64Escape(
              initialize_config.reference_values().SerializeAsString())),
          data_parser_instance.attr("parse_read_response_to_value"));
}

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee