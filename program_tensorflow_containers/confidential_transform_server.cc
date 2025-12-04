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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "absl/strings/escaping.h"

namespace confidential_federated_compute::tensorflow::program_executor_tee {

std::optional<pybind11::function>
TensorflowProgramExecutorTeeConfidentialTransform::GetProgramInitializeFn() {
  fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig
      initialize_config = GetInitializeConfig();

  return pybind11::module::import("initialize_program_tensorflow")
      .attr("get_program_initialize_fn")(
          initialize_config.outgoing_server_address(), GetWorkerBnsAddresses(),
          pybind11::bytes(absl::Base64Escape(
              initialize_config.reference_values().SerializeAsString())));
}

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee