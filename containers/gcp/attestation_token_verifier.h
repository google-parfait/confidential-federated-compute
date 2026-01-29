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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_VERIFIER_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_VERIFIER_H

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "attestation_policy.pb.h"
#include "cc/attestation/verification/attestation_verifier.h"
#include "nlohmann/json.hpp"
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/evidence.pb.h"
#include "proto/attestation/verification.pb.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/verified_jwt.h"

namespace confidential_federated_compute::gcp {

/**
 * @brief Implements policy-based attestation verification for Confidential
 * Space.
 *
 * It verifies both the signature (Intel or Google, depending on policy) and
 * a comprehensive set of identity and TCB policies provided by the user
 * via a unified policy file.
 */
class AttestationTokenVerifier
    : public oak::attestation::verification::AttestationVerifier {
 public:
  virtual ~AttestationTokenVerifier() {}

  // Oak interface stub (not used in this client-side prototype).
  virtual absl::StatusOr<oak::attestation::v1::AttestationResults> Verify(
      std::chrono::time_point<std::chrono::system_clock> now,
      const ::oak::attestation::v1::Evidence& evidence,
      const ::oak::attestation::v1::Endorsements& endorsements)
      const override = 0;

  // Main verification entry point called via FFI from the Rust session layer.
  // Verifies the JWT signature using locally loaded keys, enforces policy,
  // and extracts the public key from the nonce.
  virtual absl::StatusOr<std::string> VerifyJwt(
      absl::string_view jwt_bytes) = 0;
};

/**
 * @brief Factory function to create a configured AttestationTokenProvider.
 *
 * @param attestation_policy The attestation policy for the verifier to enforce.
 * @param jwks_payload The JWKS (Json Web Key Set) to use as a reference.
 * @param dump_jwt Whether to log the received JWT evidence for debugging
 * purposes.
 *
 * @return An initialized provider instance ready to fetch tokens.
 */
absl::StatusOr<std::unique_ptr<AttestationTokenVerifier>>
CreateAttestationTokenVerifier(const AttestationPolicy& attestation_policy,
                               const std::string& jwks_payload, bool dump_jwt);

extern "C" {
// FFI wrapper for the Rust session layer to call the C++ verifier.
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len);
}

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_ATTESTATION_TOKEN_VERIFIER_H
