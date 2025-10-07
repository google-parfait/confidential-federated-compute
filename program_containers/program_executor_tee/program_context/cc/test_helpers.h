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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_TEST_HELPERS_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_TEST_HELPERS_H

#include <string>
#include <vector>

#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

tensorflow_federated::v0::GetExecutorRequest GetExecutorRequest(
    int num_clients);

tensorflow_federated::v0::CreateValueRequest CreateIntValueRequest(
    std::string executor_id, int value);

tensorflow_federated::v0::CreateValueRequest CreateIntrinsicValueRequest(
    std::string executor_id, std::string intrinsic_uri);

tensorflow_federated::v0::CreateStructRequest CreateStructRequest(
    std::string executor_id, std::vector<std::string> ref_ids);

tensorflow_federated::v0::CreateSelectionRequest CreateSelectionRequest(
    std::string executor_id, std::string source_ref_id, int index);

tensorflow_federated::v0::CreateCallRequest CreateCallRequest(
    std::string executor_id, std::string function_ref_id,
    std::string arg_ref_id);

tensorflow_federated::v0::ComputeRequest ComputeRequest(std::string executor_id,
                                                        std::string ref_id);

tensorflow_federated::v0::DisposeRequest DisposeRequest(
    std::string executor_id, std::vector<std::string> ref_ids);

tensorflow_federated::v0::DisposeExecutorRequest DisposeExecutorRequest(
    std::string executor_id);

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_TEST_HELPERS_H
