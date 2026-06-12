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
#include "src/com/google/android/as/oss/privateinference/api/private_aratea_service.pb.h"
#include "src/com/google/android/as/oss/privateinference/service/api/private_inference.pb.h"

namespace confidential_federated_compute::private_inference {

class PiClient {
 public:
  virtual ~PiClient() = default;

  virtual absl::StatusOr<std::string> Generate(const std::string& prompt) = 0;
};

// TODO: This raw string client will be replaced by the private aratea feature
// later.
absl::StatusOr<std::unique_ptr<PiClient>> CreatePiClient(
    std::string server_address);

}  // namespace confidential_federated_compute::private_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_CLIENT_H_
