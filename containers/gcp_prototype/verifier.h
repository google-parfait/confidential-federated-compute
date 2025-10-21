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

#ifndef VERIFIER_H
#define VERIFIER_H

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "cc/attestation/verification/attestation_verifier.h"
#include "nlohmann/json.hpp"
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/evidence.pb.h"
#include "proto/attestation/verification.pb.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/verified_jwt.h"

namespace gcp_prototype {

/**
 * @brief Defines the policy requirements for validating attestation tokens.
 */
struct AttestationPolicy {
  enum class VerifierType {
    kGca,  // Google Cloud Attestation (Legacy/Optional)
    kIta,  // Intel Trust Authority (Primary RoT)
  };

  // Selects the root of trust and provider configuration.
  VerifierType verifier_type = VerifierType::kIta;

  // Configuration for hardware security features.
  bool require_debug_disabled = true;
  bool require_secboot_enabled = true;

  // TCB freshness requirements (delegated to the attestation provider).
  // Checks that the GCP software stack (kernel, CS agent, etc.) is up-to-date.
  bool require_sw_tcb_uptodate = true;
  // Checks that the underlying Intel hardware (microcode, TDX module) is
  // up-to-date.
  bool require_hw_tcb_uptodate = true;

  // Minimum date requirements for TCB.
  // Uses absl::Time to strictly enforce valid date comparisons rather than
  // relying on brittle string comparisons.
  std::optional<absl::Time> min_sw_tcb_date;
  std::optional<absl::Time> min_hw_tcb_date;

  // Identity binding requirements to ensure the workload belongs to us.
  // Requires the token to be issued for this specific GCP project.
  std::string expected_project_id = "";
  // Requires this service account to be present in the token's list.
  std::string expected_service_account = "";
  // Expected SHA-256 digest of the container image.
  std::string expected_image_digest = "";
};

/**
 * @brief Implements policy-based attestation verification for Confidential
 * Space.
 *
 * It uses Intel Trust Authority (ITA) as the primary Root of Trust,
 * verifying both the Intel signature and a comprehensive set of identity and
 * TCB policies provided by the user.
 */
class MyVerifier : public oak::attestation::verification::AttestationVerifier {
 public:
  MyVerifier();

  // Initializes the verifier by fetching JWKS based on the configured policy.
  absl::Status Initialize();

  void SkipPolicyEnforcement(bool skip);
  void SetDumpJwt(bool dump);
  void SetPolicy(const AttestationPolicy& policy);

  // Oak interface stub (not used in this client-side prototype).
  absl::StatusOr<oak::attestation::v1::AttestationResults> Verify(
      std::chrono::time_point<std::chrono::system_clock> now,
      const ::oak::attestation::v1::Evidence& evidence,
      const ::oak::attestation::v1::Endorsements& endorsements) const override;

  // Main verification entry point called via FFI from the Rust session layer.
  // Verifies the JWT signature, enforces policy, and extracts the public key
  // from the nonce.
  absl::StatusOr<std::string> VerifyJwt(absl::string_view jwt_bytes);

 private:
  // Internal configuration selected based on policy.verifier_type.
  struct VerifierConfig {
    std::string expected_issuer;
    std::string jwks_url;
    std::string expected_hw_model;
  };

  VerifierConfig internal_config_;
  AttestationPolicy client_policy_;

  std::unique_ptr<crypto::tink::JwtPublicKeyVerify> jwt_verifier_;
  bool skip_policy_enforcement_ = false;
  bool dump_jwt_ = false;

  // Struct to hold all relevant claims extracted from the token payload.
  struct ExtractedClaims {
    // Core Hardware Claims
    absl::StatusOr<std::string> hwmodel = absl::InternalError("Uninitialized");
    std::optional<bool> secboot;
    absl::StatusOr<std::string> dbgstat = absl::InternalError("Uninitialized");
    std::optional<int64_t> oemid;

    // Workload/Measurement Claim
    absl::StatusOr<std::string> image_digest =
        absl::InternalError("Uninitialized");

    // GCP Identity Claims
    std::vector<std::string> google_service_accounts;
    absl::StatusOr<std::string> gce_project_id =
        absl::InternalError("Uninitialized");

    // TCB Status Claims (delegated to ITA)
    absl::StatusOr<std::string> sw_tcb_status =
        absl::InternalError("Uninitialized");
    absl::StatusOr<std::string> hw_tcb_status =
        absl::InternalError("Uninitialized");
    // TCB Date Claims (raw strings from JSON, parsed during enforcement)
    absl::StatusOr<std::string> sw_tcb_date =
        absl::InternalError("Uninitialized");
    absl::StatusOr<std::string> hw_tcb_date =
        absl::InternalError("Uninitialized");

    // Confidential Space operational attributes (e.g., DEBUG/EXPERIMENTAL).
    std::vector<std::string> cs_support_attributes;
  };

  // Verifies signature and standard claims (issuer, audience, expiry).
  absl::StatusOr<crypto::tink::VerifiedJwt> VerifyTokenSignatureAndBasicClaims(
      absl::string_view jwt_bytes) const;

  // Parses the verified JWT payload into a JSON object.
  absl::StatusOr<nlohmann::json> ParsePayload(
      crypto::tink::VerifiedJwt& verified_jwt) const;

  // Extracts key claims into the ExtractedClaims struct and logs them.
  absl::StatusOr<ExtractedClaims> ExtractAndLogClaims(
      const crypto::tink::VerifiedJwt& verified_jwt,
      const nlohmann::json& payload_json) const;

  // Enforces all policy rules against the extracted claims.
  absl::Status EnforcePolicy(const ExtractedClaims& claims) const;

  // Extracts and Base64-decodes the attestation public key from the 'eat_nonce'
  // claim.
  absl::StatusOr<std::string> ExtractNonce(
      const nlohmann::json& payload_json) const;

  // Helper for enforcing exact string matches in policy to reduce boilerplate.
  absl::Status CheckStringMatch(const absl::StatusOr<std::string>& actual,
                                const std::string& expected,
                                absl::string_view claim_label) const;
};

}  // namespace gcp_prototype

extern "C" {
// FFI wrapper for the Rust session layer to call the C++ verifier.
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len);
}

#endif  // VERIFIER_H
