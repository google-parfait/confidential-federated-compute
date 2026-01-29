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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_PROVIDER_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_PROVIDER_H

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::gcp {

/**
 * @brief Abstract interface for fetching a signed attestation token from the
 * Confidential Space agent.
 */
class AttestationTokenProvider {
 public:
  virtual ~AttestationTokenProvider() = default;

  /**
   * @brief Fetches an attestation token, incorporating the provided nonce.
   *
   * @param nonce A high-entropy string (typically a Base64-encoded public key)
   * to be bound to the attestation token to prevent replay attacks.
   * @return The raw JWT token string on success, or an error status.
   */
  virtual absl::StatusOr<std::string> GetAttestationToken(
      absl::string_view nonce) = 0;
};

/**
 * @brief Enumeration of supported attestation authorities.
 */
enum class ProviderType {
  kGca,  // Google Cloud Attestation (Uses OIDC token type).
  kIta,  // Intel Trust Authority (Uses PRINCIPAL_TAGS token type).
};

/**
 * @brief Factory function to create a configured AttestationTokenProvider.
 *
 * @param type The desired provider authority.
 * @return An initialized provider instance ready to fetch tokens.
 */
std::unique_ptr<AttestationTokenProvider> CreateAttestationTokenProvider(
    ProviderType type);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_PROVIDER_H
