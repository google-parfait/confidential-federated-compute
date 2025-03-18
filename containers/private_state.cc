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

#include "containers/private_state.h"

#include <memory>
#include <utility>

#include "fcp/base/monitoring.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace confidential_federated_compute {

absl::StatusOr<std::unique_ptr<PrivateState>> PrivateState::Parse(
    google::protobuf::io::CodedInputStream* stream) {
  return std::make_unique<PrivateState>();
}

void PrivateState::Serialize(
    google::protobuf::io::CodedOutputStream* stream) const {}

absl::Status PrivateState::Merge(const PrivateState& other) {
  return absl::OkStatus();
}

std::string BundlePrivateState(const std::string& blob,
                               const PrivateState& private_state) {
  std::string buffer;
  google::protobuf::io::StringOutputStream stream(&buffer);
  google::protobuf::io::CodedOutputStream coded_stream(&stream);
  coded_stream.WriteString(kPrivateStateBundleSignature);
  private_state.Serialize(&coded_stream);
  coded_stream.WriteVarint64(blob.size());
  coded_stream.Trim();
  buffer.append(blob);
  return buffer;
}

// Extracts private state from a blob and replaces the blob in place.
absl::StatusOr<std::unique_ptr<PrivateState>> UnbundlePrivateState(
    std::string& blob) {
  google::protobuf::io::ArrayInputStream stream(blob.data(), blob.size());
  google::protobuf::io::CodedInputStream coded_stream(&stream);
  std::string signature;
  if (!coded_stream.ReadString(&signature,
                               sizeof(kPrivateStateBundleSignature) - 1) ||
      signature != kPrivateStateBundleSignature) {
    return absl::InvalidArgumentError(
        "Invalid input blob: private state bundle signature mismatch");
  }

  FCP_ASSIGN_OR_RETURN(std::unique_ptr<PrivateState> private_state,
                       PrivateState::Parse(&coded_stream));

  size_t payload_size;
  if (!coded_stream.ReadVarint64(&payload_size)) {
    return absl::InvalidArgumentError(
        "Invalid input blob: payload size is missing");
  }

  int pos = coded_stream.CurrentPosition();

  if (pos + payload_size != blob.size()) {
    return absl::InvalidArgumentError("Invalid input blob: incomplete payload");
  }

  // Make a new blob for the remaining "payload" part of the original blob.
  blob = blob.substr(pos);
  return private_state;
}

}  // namespace confidential_federated_compute
