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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CHECKPOINT_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CHECKPOINT_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {

// Represents the extracted checkpoint metadata (privacy ID) and 1-dimensional
// column tensors (event time and client data columns) from a client upload.
//
// This class validates and maintains the required tensor structure for
// construct sessions, ensuring that metadata and column tensors are properly
// formatted and aligned in dimension.
class Checkpoint {
 public:
  // Returns a string view of the privacy ID.
  absl::string_view privacy_id() const;

  // Moves and returns the original scalar privacy ID tensor. This method is
  // destructive.
  tensorflow_federated::aggregation::Tensor take_privacy_id_tensor();

  // Returns a map of all 1-dimensional column tensors, including the event time
  // DT_STRING tensor and opaque client data columns.
  const absl::flat_hash_map<std::string,
                            tensorflow_federated::aggregation::Tensor>&
  column_tensors() const;

  // Validates that the contents of `checkpoint` match the expected
  // structure, including the presence and formatting of checkpoint metadata and
  // column tensors, and constructs a Checkpoint upon success.
  static absl::StatusOr<Checkpoint> Create(
      tensorflow_federated::aggregation::CheckpointParser& checkpoint,
      absl::string_view on_device_query_name);

  Checkpoint(Checkpoint&&) = default;
  Checkpoint& operator=(Checkpoint&&) = default;
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;

 private:
  Checkpoint(tensorflow_federated::aggregation::Tensor privacy_id_tensor,
             absl::flat_hash_map<std::string,
                                 tensorflow_federated::aggregation::Tensor>
                 column_tensors);

  // Original scalar DT_STRING tensor for the privacy ID.
  tensorflow_federated::aggregation::Tensor privacy_id_tensor_;

  // Map of tensor name to tensor object. Includes all 1-dimensional column
  // tensors (event time and client data columns).
  //
  // Invariants:
  // - The event time tensor is a 1-dimensional DT_STRING tensor.
  // - All other column tensors are 1-dimensional with the same row count as the
  //   event time tensor.
  absl::flat_hash_map<std::string, tensorflow_federated::aggregation::Tensor>
      column_tensors_;
};

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CHECKPOINT_H_
