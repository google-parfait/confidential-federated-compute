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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_CLIENT_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_CLIENT_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "proto/private_aratea_service.pb.h"

namespace confidential_federated_compute::private_inference {

class PiClient {
 public:
  virtual ~PiClient() = default;

  virtual absl::StatusOr<std::string> Generate(const std::string& prompt) = 0;
};

// Creates a PiClient.
//
// The `feature_name` is configured via the client in their pipeline config and
// passed via Flume to this container. It is used to select the model and track
// usage. The list of feature names is defined in
// `private_aratea_service.proto`.
absl::StatusOr<std::unique_ptr<PiClient>> CreatePiClient(
    std::string server_address,
    ::private_inference::proto::FeatureName feature_name);

}  // namespace confidential_federated_compute::private_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_CLIENT_H_
