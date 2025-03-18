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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_STATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_STATE_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/io/coded_stream.h"

namespace confidential_federated_compute {

// Signature written in the beginning of the bundled blob that
// contains the private state.
inline constexpr const char kPrivateStateBundleSignature[] = "PSv1";

// Confidential Transform Session private state.
class PrivateState {
 public:
  // Parses the private state.
  static absl::StatusOr<std::unique_ptr<PrivateState>> Parse(
      google::protobuf::io::CodedInputStream* stream);

  // Serializes the private state.
  void Serialize(google::protobuf::io::CodedOutputStream* stream) const;

  // Merges the private state with another private state.
  absl::Status Merge(const PrivateState& other);
};

// Attaches private state to a blob and returns a combined blob.
std::string BundlePrivateState(const std::string& blob,
                               const PrivateState& private_state);

// Extracts private state from a blob and replaces the blob in place.
absl::StatusOr<std::unique_ptr<PrivateState>> UnbundlePrivateState(
    std::string& blob);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_STATE_H_
