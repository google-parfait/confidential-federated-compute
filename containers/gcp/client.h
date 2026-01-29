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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_H

#include <functional>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "attestation_token_verifier.h"

namespace confidential_federated_compute::gcp {

// Abstract interface for the generic GCP client.
class Client {
 public:
  virtual ~Client() {}

  // Sends a string request to the server snd returns a string response.
  virtual absl::StatusOr<std::string> Invoke(std::string request) = 0;
};

// Constructs a client that performs attestation, can send string
// requests to the server, and receive string payloads. Specific
// types of client-server protocols can be layered over this, with
// specialized request and response protobuf structues passed in a
// string-serialized form.
absl::StatusOr<std::unique_ptr<Client>> CreateClient(
    std::string server_address,
    std::unique_ptr<AttestationTokenVerifier> attestation_token_verifier);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_H
