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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_SERVER_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_SERVER_H

#include <functional>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "attestation_token_provider.h"

namespace confidential_federated_compute::gcp {

class Server {
 public:
  using RequestHandler =
      absl::AnyInvocable<absl::StatusOr<std::string>(std::string)>;
  virtual ~Server() {}
  virtual void Wait() = 0;
  virtual int port() = 0;
};

// Constructs a server that lives on the specified port, uses the
// specified handler to parse request strings and produce string
// responses, and handles attestation requests using the gspecified
// attestation provider. Specialized services can be layered on top
// of this by passing specific request/response protobufs in a
// string-serialized form and by supplying a handler that knows how
// to unpack and handle them.
absl::StatusOr<std::unique_ptr<Server>> CreateServer(
    int port,
    std::unique_ptr<AttestationTokenProvider> attestation_token_provider,
    Server::RequestHandler request_handler);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_SERVER_H
