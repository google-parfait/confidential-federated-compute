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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_XLA_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_XLA_H_

#include <pybind11/functional.h>

#include <memory>
#include <optional>

#include "cc/crypto/signing_key.h"
#include "program_executor_tee/confidential_transform_server.h"

namespace confidential_federated_compute::program_executor_tee {

// ConfidentialTransform service for XLA program executor TEE.
class XLAProgramExecutorTeeConfidentialTransform final
    : public confidential_federated_compute::program_executor_tee::
          ProgramExecutorTeeConfidentialTransform {
 public:
  XLAProgramExecutorTeeConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle)
      : ProgramExecutorTeeConfidentialTransform(std::move(signing_key_handle)) {
  }

  std::optional<pybind11::function> GetProgramInitializeFn() override;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_CONFIDENTIAL_TRANSFORM_SERVER_XLA_H_
