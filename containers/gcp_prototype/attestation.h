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

// Header for the helper function to fetch attestation tokens from the
// GCP Confidential Space agent.

#ifndef ATTESTATION_H
#define ATTESTATION_H

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace gcp_prototype {

/**
 * @brief Fetches an OIDC attestation token from the local Confidential Space
 * agent.
 *
 * Sends an HTTP POST request over a Unix domain socket to the agent's token
 * endpoint (`/v1/token`). The request includes the specified audience
 * ("oak_session_noise_v1") and the provided nonce (which should be the
 * Base64-encoded public key).
 *
 * @param nonce The Base64-encoded nonce to include in the token request.
 * @return On success, the raw JWT attestation token string. On failure, an
 * error status explaining the issue (e.g., curl error, HTTP error from agent).
 */
absl::StatusOr<std::string> GetAttestationToken(absl::string_view nonce);

}  // namespace gcp_prototype

#endif  // ATTESTATION_H
