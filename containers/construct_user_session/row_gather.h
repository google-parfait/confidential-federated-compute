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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_ROW_GATHER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_ROW_GATHER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "containers/construct_user_session/checkpoint.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace confidential_federated_compute::construct_user_session {

// Filters and gathers rows from a group of Checkpoints. For each checkpoint,
// event times are filtered against the session window [window_start,
// window_end). Surviving rows are gathered across all checkpoints and
// concatenated into output tensors.
//
// Returns a map of column name to merged tensor. Columns that are empty are
// omitted from the result.
//
// IMPORTANT: `checkpoints` must outlive the returned tensors (the output
// may contain string_view references into the source checkpoint tensors).
absl::flat_hash_map<std::string, tensorflow_federated::aggregation::Tensor>
GatherSessionRows(absl::Span<const Checkpoint> checkpoints,
                  absl::string_view event_time_tensor_name,
                  absl::Time window_start, absl::Time window_end,
                  const absl::flat_hash_map<
                      std::string, tensorflow_federated::aggregation::DataType>&
                      column_dtypes);

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_ROW_GATHER_H_
