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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_METADATA_MAP_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_METADATA_MAP_FN_H_

#include <optional>
#include <string>
#include <vector>

#include "containers/fns/map_fn.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"

namespace confidential_federated_compute::metadata {

// MapFn implementation that computes metadata for each input.
// Not thread-safe.
class MetadataMapFn final : public confidential_federated_compute::fns::MapFn {
 public:
  explicit MetadataMapFn(
      const fcp::confidentialcompute::MetadataContainerConfig& config)
      : config_(config) {};

  // Parses the unencrypted data, for each metadata config compute the
  // corresponding metadata.
  absl::StatusOr<confidential_federated_compute::fns::KeyValue> Map(
      confidential_federated_compute::fns::KeyValue input,
      Context& context) override;

 private:
  const fcp::confidentialcompute::MetadataContainerConfig config_;
};
}  // namespace confidential_federated_compute::metadata

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_METADATA_MAP_FN_H_
