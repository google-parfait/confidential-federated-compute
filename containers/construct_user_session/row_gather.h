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
#include "absl/types/span.h"
#include "containers/common/row_set.h"
#include "containers/construct_user_session/ingestion.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace confidential_federated_compute::construct_user_session {

// Gathers all surviving rows for a group of row locations (typically
// corresponding to a single privacy ID), returning column tensors
// in a single map. `group` must be sorted.
//
// IMPORTANT: `inputs` must outlive the returned tensors.
absl::flat_hash_map<std::string, tensorflow_federated::aggregation::Tensor>
GatherSurvivingRows(
    absl::Span<const RowLocation> group, absl::Span<const Checkpoint> inputs,
    const absl::flat_hash_map<std::string,
                              tensorflow_federated::aggregation::DataType>&
        column_dtypes);

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_ROW_GATHER_H_
