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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_ELASTIC_COMPOSING_EXECUTOR_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_ELASTIC_COMPOSING_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace confidential_federated_compute::program_executor_tee {

// ElasticComposingExecutor dynamically distributes per-client work across child
// worker executors using a shared work queue, rather than statically
// partitioning clients across children at value-creation time.
//
// Core properties:
//   - Batch-based distribution: clients are grouped into batches and
//     distributed to workers as federated intrinsics (e.g. federated_map,
//     federated_aggregate), amortising RPC overhead while retaining
//     dynamic load balancing flexibility.
//   - Fault tolerance: if a child executor fails mid-round, unfinished work is
//     re-queued and picked up by remaining healthy children.
std::shared_ptr<tensorflow_federated::Executor> CreateElasticComposingExecutor(
    std::shared_ptr<tensorflow_federated::Executor> server,
    std::vector<tensorflow_federated::ComposingChild> children,
    int32_t total_clients, int32_t avg_batches_per_worker = 10);

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_ELASTIC_COMPOSING_EXECUTOR_H_
