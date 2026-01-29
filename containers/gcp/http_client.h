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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_HTTP_CLIENT_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_HTTP_CLIENT_H

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::gcp {

// Performs an HTTP POST request with a JSON payload
// over a Unix domain socket using libcurl.
absl::StatusOr<std::string> PostJsonViaUnixSocket(
    absl::string_view url, absl::string_view socket_path,
    absl::string_view json_payload);

// Performs a standard HTTP GET request using libcurl.
absl::StatusOr<std::string> CurlGet(absl::string_view url);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_HTTP_CLIENT_H
