// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef VERIFIER_H
#define VERIFIER_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cc/attestation/verification/attestation_verifier.h"
#include "nlohmann/json.hpp"  // For nlohmann::json
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/evidence.pb.h"
#include "proto/attestation/verification.pb.h"
#include "proto/session/messages.pb.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/verified_jwt.h"  // For crypto::tink::VerifiedJwt

namespace gcp_prototype {

/**
 * @brief Defines the policy requirements for validating Confidential Space
 * attestation tokens.
 *
 * Future work (b/452094015): Incorporate the verification of the system image
 * and boot proceess, possibly via Intel Trust Authority (ITA).
 */
struct AttestationPolicy {
  /**
   * @brief If true, requires the 'dbgstat' claim in the attestation token to
   * be exactly "disabled". Default is true (for production readiness).
   */
  bool require_debug_disabled = true;

  /**
   * @brief If true, requires the 'secboot' claim in the attestation token to
   * be true. Default is true.
   */
  bool require_secboot_enabled = true;

  /**
   * @brief Expected value for the 'hwmodel' claim (e.g., "GCP_INTEL_TDX").
   * Default requires Intel TDX on GCP.
   */
  std::string expected_hw_model = "GCP_INTEL_TDX";

  /**
   * @brief Expected SHA-256 digest of the container image config blob
   * (e.g., "sha256:..."). If empty, the check is skipped.
   */
  std::string expected_image_digest = "";
  // Future work (b/452094015): Add policy fields for swversion or other claims
  // if needed.
};

/**
 * @brief Implements attestation verification logic for GCP Confidential Space.
 *
 * This class verifies Confidential Space attestation tokens (JWTs) against a
 * defined policy and handles the extraction of the public key nonce used for
 * session binding.
 *
 * It implements the Oak `AttestationVerifier` interface but currently only
 * provides a stub implementation for the `Verify` method (which handles
 * endorsements). The primary logic resides in the `VerifyJwt` method, called
 * via FFI from Rust during the Oak handshake.
 *
 * Future work (b/452094015): Implement endorsement verification in the `Verify`
 * method, potentially by integrating with Intel Trust Authority (ITA). Future
 * work (b/452094015): Implement root-of-trust anchoring beyond trusting the
 * JWKS URL fetched during initialization.
 */
class MyVerifier : public oak::attestation::verification::AttestationVerifier {
 public:
  MyVerifier();

  /**
   * @brief Initializes the verifier by fetching Google's public keys (JWKS).
   * Must be called before `VerifyJwt` or the FFI wrapper.
   * @return absl::OkStatus() on success, error status otherwise.
   */
  absl::Status Initialize();

  /**
   * @brief Sets the attestation policy to be enforced by `VerifyJwt`.
   * @param policy The policy configuration.
   */
  void SetPolicy(const AttestationPolicy& policy);

  /**
   * @brief Implements the Oak AttestationVerifier interface (currently a stub).
   * This method is intended for verifying endorsements associated with the
   * evidence.
   * @param now Current time point.
   * @param evidence Attestation evidence containing the JWT.
   * @param endorsements Endorsements providing trust in the hardware/platform.
   * @return Currently always returns a success status with dummy results.
   */
  absl::StatusOr<oak::attestation::v1::AttestationResults> Verify(
      std::chrono::time_point<std::chrono::system_clock> now,
      const ::oak::attestation::v1::Evidence& evidence,
      const ::oak::attestation::v1::Endorsements& endorsements) const override;

  /**
   * @brief Verifies a GCP Confidential Space JWT and extracts the public key
   * nonce.
   *
   * This method performs the following checks:
   * 1. Verifies the JWT signature using Google's public keys.
   * 2. Verifies standard claims (issuer, audience, type, timestamps) via Tink.
   * 3. Parses the JWT payload.
   * 4. Extracts and logs relevant Confidential Computing claims.
   * 5. Enforces the configured AttestationPolicy against these claims.
   * 6. Extracts, decodes, and returns the public key from the 'eat_nonce'
   * claim.
   *
   * @param jwt_bytes The raw JWT bytes received from the server.
   * @return On success, the raw bytes of the public key extracted from the
   * nonce. On failure, an error status explaining the reason (e.g., signature
   * mismatch, policy violation).
   */
  absl::StatusOr<std::string> VerifyJwt(absl::string_view jwt_bytes);

 private:
  // Tink primitive for verifying JWT signatures.
  std::unique_ptr<crypto::tink::JwtPublicKeyVerify> jwt_verifier_;
  // Policy configuration to enforce.
  AttestationPolicy policy_;

  /**
   * @brief Helper struct to hold extracted claims for policy evaluation.
   * Uses StatusOr/optional to handle cases where claims might be missing or
   * extraction fails.
   */
  struct ExtractedClaims {
    // --- FIX ---
    // Provide explicit in-class initializers for StatusOr members.
    // This makes the struct default-constructible.
    absl::StatusOr<std::string> hwmodel = absl::InternalError("Uninitialized");
    std::optional<bool> secboot;  // optionals are fine to default construct
    absl::StatusOr<std::string> dbgstat = absl::InternalError("Uninitialized");
    std::optional<int64_t> oemid;
    absl::StatusOr<std::string> image_digest =
        absl::InternalError("Uninitialized");
    // --- END FIX ---
  };

  // Helper methods for VerifyJwt logic.
  absl::StatusOr<crypto::tink::VerifiedJwt> VerifyTokenSignatureAndBasicClaims(
      absl::string_view jwt_bytes) const;
  absl::StatusOr<nlohmann::json> ParsePayload(
      crypto::tink::VerifiedJwt& verified_jwt) const;
  absl::StatusOr<ExtractedClaims> ExtractAndLogClaims(
      const crypto::tink::VerifiedJwt& verified_jwt,
      const nlohmann::json& payload_json) const;
  absl::Status EnforcePolicy(const ExtractedClaims& claims) const;
  absl::StatusOr<std::string> ExtractNonce(
      const crypto::tink::VerifiedJwt& verified_jwt) const;
};

}  // namespace gcp_prototype

// --- C FFI Wrapper ---

// Exposes the `MyVerifier::VerifyJwt` functionality to Rust code.
// See `MyVerifier::VerifyJwt` for detailed verification steps.
extern "C" {
/**
 * @brief C-style wrapper function to verify a JWT and extract the nonce public
 * key. Called by Rust via FFI.
 *
 * @param context Opaque pointer to the MyVerifier instance.
 * @param jwt_bytes Pointer to the JWT data.
 * @param jwt_len Length of the JWT data.
 * @param out_public_key Output buffer to write the extracted public key bytes.
 * @param out_public_key_len In/out parameter. Input: capacity of
 * `out_public_key`. Output: actual size of the written public key, or the
 * required size if the buffer was too small.
 * @return true on successful verification and extraction, false otherwise
 * (e.g., verification failure, policy violation, buffer too small). Error
 * details are logged internally.
 */
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len);
}  // extern "C"

#endif  // VERIFIER_H
