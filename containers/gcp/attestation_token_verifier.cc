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

#include "attestation_token_verifier.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "attestation_policy.pb.h"
#include "http_client.h"
#include "json_parsing_utils.h"
#include "nlohmann/json.hpp"
#include "proto/attestation/verification.pb.h"
#include "tink/config/global_registry.h"
#include "tink/jwt/jwk_set_converter.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/jwt_signature_config.h"
#include "tink/jwt/jwt_validator.h"
#include "tink/jwt/verified_jwt.h"
#include "tink/util/statusor.h"

namespace confidential_federated_compute::gcp {
namespace {

// Standard JWT Claims used in the attestation token.
constexpr char kAudience[] = "oak_session_noise_v1";
constexpr char kJwtNonceAttributeName[] = "eat_nonce";
constexpr char kJwtHwModelAttributeName[] = "hwmodel";
constexpr char kJwtSecbootAttributeName[] = "secboot";
constexpr char kJwtDbgstatAttributeName[] = "dbgstat";
constexpr char kJwtSubmodsAttributeName[] = "submods";
constexpr char kSubmodsContainerFieldName[] = "container";
constexpr char kContainerImageDigestFieldName[] = "image_digest";
constexpr char kSwVersionAttributeName[] = "swversion";
constexpr char kOemIdAttributeName[] = "oemid";

// Deeply nested claims used for identity and TCB status verification.
constexpr char kGoogleServiceAccounts[] = "google_service_accounts";
constexpr char kSubmodGce[] = "gce";
constexpr char kGceProjectId[] = "project_id";
constexpr char kTdx[] = "tdx";
constexpr char kSwTcbStatus[] = "gcp_attester_tcb_status";
constexpr char kSwTcbDate[] = "gcp_attester_tcb_date";
constexpr char kHwTcbStatus[] = "attester_tcb_status";
constexpr char kHwTcbDate[] = "attester_tcb_date";
constexpr char kSubmodConfidentialSpace[] = "confidential_space";
constexpr char kSupportAttributes[] = "support_attributes";

/**
 * @brief Implements policy-based attestation verification for Confidential
 * Space.
 *
 * It verifies both the signature (Intel or Google, depending on policy) and
 * a comprehensive set of identity and TCB policies provided by the user
 * via a unified policy file.
 */
class AttestationTokenVerifierImpl : public AttestationTokenVerifier {
 public:
  AttestationTokenVerifierImpl(
      const AttestationPolicy& attestation_policy,
      std::unique_ptr<crypto::tink::JwtPublicKeyVerify> jwt_verifier,
      bool dump_jwt);

  ~AttestationTokenVerifierImpl() {}

  // Oak interface stub (not used in this client-side prototype).
  virtual absl::StatusOr<oak::attestation::v1::AttestationResults> Verify(
      std::chrono::time_point<std::chrono::system_clock> now,
      const ::oak::attestation::v1::Evidence& evidence,
      const ::oak::attestation::v1::Endorsements& endorsements) const override;

  // Main verification entry point called via FFI from the Rust session layer.
  // Verifies the JWT signature using locally loaded keys, enforces policy,
  // and extracts the public key from the nonce.
  virtual absl::StatusOr<std::string> VerifyJwt(
      absl::string_view jwt_bytes) override;

 private:
  // Configuration for the selected Root of Trust (ITA or GCA).
  struct VerifierConfig {
    std::string expected_issuer;
    std::string expected_hw_model;
  };

  VerifierConfig internal_config_;
  AttestationPolicy policy_;

  std::unique_ptr<crypto::tink::JwtPublicKeyVerify> jwt_verifier_;
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

    // TCB Status Claims (delegated to ITA, or GCA equivalent)
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

AttestationTokenVerifierImpl::AttestationTokenVerifierImpl(
    const AttestationPolicy& attestation_policy,
    std::unique_ptr<crypto::tink::JwtPublicKeyVerify> jwt_verifier,
    bool dump_jwt)
    : policy_(attestation_policy),
      jwt_verifier_(std::move(jwt_verifier)),
      dump_jwt_(dump_jwt) {
  switch (policy_.verifier_type()) {
    case AttestationPolicy::GCA:
      internal_config_ = {
          .expected_issuer = "https://confidentialcomputing.googleapis.com",
          .expected_hw_model = "GCP_INTEL_TDX"};
      break;
    case AttestationPolicy::ITA:
    default:  // Default to ITA if unspecified.
      internal_config_ = {
          .expected_issuer = "https://portal.trustauthority.intel.com",
          .expected_hw_model = "INTEL_TDX"};
      break;
  }
  LOG(INFO) << "Attestation policy:\n" << policy_.DebugString();
  LOG(INFO) << "Internal verifier config set: expected_issuer="
            << internal_config_.expected_issuer
            << ", expected_hw_model=" << internal_config_.expected_hw_model;
  LOG(INFO) << "JWT Verifier initialized successfully.";
}

absl::StatusOr<oak::attestation::v1::AttestationResults>
AttestationTokenVerifierImpl::Verify(
    std::chrono::time_point<std::chrono::system_clock> now,
    const ::oak::attestation::v1::Evidence& evidence,
    const ::oak::attestation::v1::Endorsements& endorsements) const {
  // This method will remain unimplemented. Evidence and Endorsements are older
  // data structures that are irrelevant in this context (we use assertions).
  LOG(WARNING)
      << "MyVerifier::Verify is NOT IMPLEMENTED (returning success stub).";
  oak::attestation::v1::AttestationResults results;
  results.set_status(oak::attestation::v1::AttestationResults::STATUS_SUCCESS);
  return results;
}

absl::StatusOr<std::string> AttestationTokenVerifierImpl::VerifyJwt(
    absl::string_view jwt_bytes) {
  LOG(INFO) << "C++ MyVerifier::VerifyJwt called with token (size "
            << jwt_bytes.size() << ").";

  if (!jwt_verifier_) {
    return absl::FailedPreconditionError(
        "JWT Verifier not initialized. Call Initialize() with a valid JWKS "
        "path "
        "first.");
  }

  // 1. Verify the signature and basic claims (issuer/audience).
  absl::StatusOr<crypto::tink::VerifiedJwt> verified_jwt_or =
      VerifyTokenSignatureAndBasicClaims(jwt_bytes);
  if (!verified_jwt_or.ok()) {
    return verified_jwt_or.status();
  }
  crypto::tink::VerifiedJwt& verified_jwt = *verified_jwt_or;

  // 2. Parse the payload for deep inspection.
  absl::StatusOr<nlohmann::json> payload_json_or = ParsePayload(verified_jwt);
  if (!payload_json_or.ok()) {
    return payload_json_or.status();
  }
  const nlohmann::json& payload_json = *payload_json_or;

  if (dump_jwt_) {
    LOG(INFO) << "--- BEGIN JWT PAYLOAD DUMP ---";
    try {
      LOG(INFO) << payload_json.dump(2);
    } catch (const nlohmann::json::exception& e) {
      LOG(ERROR) << "Failed to dump JSON payload: " << e.what();
    }
    LOG(INFO) << "--- END JWT PAYLOAD DUMP ---";
  }

  // 3. Extract and log all relevant claims.
  absl::StatusOr<ExtractedClaims> claims_or =
      ExtractAndLogClaims(verified_jwt, payload_json);
  if (!claims_or.ok()) {
    LOG(ERROR) << "Failed to extract some claims, proceeding to policy check: "
               << claims_or.status();
  }
  const ExtractedClaims& claims = claims_or.value_or(ExtractedClaims{});

  // 4. Enforce the security policy.
  absl::Status policy_status = EnforcePolicy(claims);
  if (!policy_status.ok()) {
    return policy_status;
  }

  // 5. Extract the attestation public key from the nonce claim.
  return ExtractNonce(payload_json);
}

absl::StatusOr<crypto::tink::VerifiedJwt>
AttestationTokenVerifierImpl::VerifyTokenSignatureAndBasicClaims(
    absl::string_view jwt_bytes) const {
  // Configure validator for standard required claims.
  absl::StatusOr<crypto::tink::JwtValidator> validator_or =
      crypto::tink::JwtValidatorBuilder()
          .ExpectTypeHeader("JWT")
          .ExpectIssuer(internal_config_.expected_issuer)
          .ExpectAudience(kAudience)
          .SetClockSkew(absl::Minutes(5))  // Allow for minor clock skew
          .Build();
  if (!validator_or.ok()) {
    return absl::InternalError(absl::StrCat("Failed to build JWT validator: ",
                                            validator_or.status().ToString()));
  }

  // Verify signature using JWKS and decode the token while validating claims.
  absl::StatusOr<crypto::tink::VerifiedJwt> verified_jwt_or =
      jwt_verifier_->VerifyAndDecode(std::string(jwt_bytes), *validator_or);
  if (!verified_jwt_or.ok()) {
    return absl::InternalError(
        absl::StrCat("JWT signature/basic claim verification failed: ",
                     verified_jwt_or.status().ToString()));
  }
  LOG(INFO) << "JWT signature and basic claims verified successfully.";
  return verified_jwt_or;
}

absl::StatusOr<nlohmann::json> AttestationTokenVerifierImpl::ParsePayload(
    crypto::tink::VerifiedJwt& verified_jwt) const {
  absl::StatusOr<std::string> payload_str_or = verified_jwt.GetJsonPayload();
  if (!payload_str_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to get JWT JSON payload: ",
                     payload_str_or.status().ToString()));
  }
  try {
    return nlohmann::json::parse(*payload_str_or);
  } catch (const nlohmann::json::parse_error& e) {
    return absl::InternalError(
        absl::StrCat("Failed to parse JWT JSON payload: ", e.what()));
  }
}

absl::StatusOr<AttestationTokenVerifierImpl::ExtractedClaims>
AttestationTokenVerifierImpl::ExtractAndLogClaims(
    const crypto::tink::VerifiedJwt& verified_jwt,
    const nlohmann::json& payload_json) const {
  ExtractedClaims claims;
  LOG(INFO) << "--- Extracted JWT Claims ---";

  // Helper lambda to log string claims.
  auto log_parsed_claim = [&](absl::string_view claim_name,
                              const absl::StatusOr<std::string>& claim_or) {
    if (claim_or.ok()) {
      LOG(INFO) << "  " << claim_name << ": " << *claim_or;
    } else if (claim_or.status().code() == absl::StatusCode::kNotFound) {
      LOG(WARNING) << "  " << claim_name << ": Claim not found.";
    } else {
      LOG(ERROR) << "  Failed to get claim " << claim_name << ": "
                 << claim_or.status();
    }
  };
  // Helper lambda to log boolean claims from VerifiedJwt.
  auto log_tink_bool_claim =
      [&](absl::string_view claim_name) -> std::optional<bool> {
    auto claim_or = verified_jwt.GetBooleanClaim(std::string(claim_name));
    if (claim_or.ok()) {
      LOG(INFO) << "  " << claim_name << ": " << (*claim_or ? "true" : "false");
      return *claim_or;
    } else {
      LOG(WARNING) << "  " << claim_name << ": Claim not found or not boolean.";
      return std::nullopt;
    }
  };
  // Helper lambda to log integer claims from VerifiedJwt.
  auto log_tink_int_claim =
      [&](absl::string_view claim_name) -> std::optional<int64_t> {
    auto claim_or = verified_jwt.GetNumberClaim(std::string(claim_name));
    if (claim_or.ok()) {
      double val = *claim_or;
      double intpart;
      if (std::modf(val, &intpart) == 0.0) {
        LOG(INFO) << "  " << claim_name << ": " << static_cast<int64_t>(val);
        return static_cast<int64_t>(val);
      } else {
        LOG(WARNING) << "  " << claim_name
                     << ": Claim found but is not an integer (" << val << ").";
        return std::nullopt;
      }
    } else {
      LOG(WARNING) << "  " << claim_name << ": Claim not found or not numeric.";
      return std::nullopt;
    }
  };

  // 1. Core Hardware and Security Claims
  claims.hwmodel =
      GetStringClaimFromPath(payload_json, {kJwtHwModelAttributeName});
  log_parsed_claim(kJwtHwModelAttributeName, claims.hwmodel);

  claims.secboot = log_tink_bool_claim(kJwtSecbootAttributeName);

  claims.dbgstat =
      GetStringClaimFromPath(payload_json, {kJwtDbgstatAttributeName});
  log_parsed_claim(kJwtDbgstatAttributeName, claims.dbgstat);

  // Log informational claims (not enforced by current policy).
  if (payload_json.contains(kSwVersionAttributeName)) {
    LOG(INFO) << "  " << kSwVersionAttributeName << ": "
              << payload_json.at(kSwVersionAttributeName).dump();
  } else {
    LOG(WARNING) << "  " << kSwVersionAttributeName << ": Claim not found.";
  }

  claims.oemid = log_tink_int_claim(kOemIdAttributeName);

  // 2. Workload Measurement (Container Image Digest)
  claims.image_digest = GetStringClaimFromPath(
      payload_json, {kJwtSubmodsAttributeName, kSubmodsContainerFieldName,
                     kContainerImageDigestFieldName});
  log_parsed_claim("image_digest", claims.image_digest);

  // 3. GCP Identity Claims
  auto sa_list_or =
      GetStringListClaimFromPath(payload_json, {kGoogleServiceAccounts});
  if (sa_list_or.ok()) {
    claims.google_service_accounts = *sa_list_or;
  } else {
    LOG(WARNING) << "Failed to parse google_service_accounts as string list: "
                 << sa_list_or.status();
  }
  LOG(INFO) << "  google_service_accounts: "
            << (claims.google_service_accounts.empty()
                    ? "[None Found]"
                    : nlohmann::json(claims.google_service_accounts).dump());

  // Extract GCP Project ID.
  claims.gce_project_id = GetStringClaimFromPath(
      payload_json, {kJwtSubmodsAttributeName, kSubmodGce, kGceProjectId});
  log_parsed_claim("gce.project_id", claims.gce_project_id);

  // 4. TCB Statuses (Delegated Intel Verification)
  // GCP Software TCB Status (kernel, CS agent, etc.).
  claims.sw_tcb_status =
      GetStringClaimFromPath(payload_json, {kTdx, kSwTcbStatus});
  log_parsed_claim("tdx.gcp_attester_tcb_status (SW)", claims.sw_tcb_status);

  claims.sw_tcb_date = GetStringClaimFromPath(payload_json, {kTdx, kSwTcbDate});
  log_parsed_claim("tdx.gcp_attester_tcb_date (SW)", claims.sw_tcb_date);

  // Intel Hardware TCB Status (microcode, TDX module).
  claims.hw_tcb_status =
      GetStringClaimFromPath(payload_json, {kTdx, kHwTcbStatus});
  log_parsed_claim("tdx.attester_tcb_status (HW)", claims.hw_tcb_status);

  claims.hw_tcb_date = GetStringClaimFromPath(payload_json, {kTdx, kHwTcbDate});
  log_parsed_claim("tdx.attester_tcb_date (HW)", claims.hw_tcb_date);

  // 5. Confidential Space Support Attributes (e.g., DEBUG flag)
  auto support_attrs_or = GetStringListClaimFromPath(
      payload_json,
      {kJwtSubmodsAttributeName, kSubmodConfidentialSpace, kSupportAttributes});
  if (support_attrs_or.ok()) {
    claims.cs_support_attributes = *support_attrs_or;
  } else {
    LOG(WARNING) << "Failed to parse support_attributes as string list: "
                 << support_attrs_or.status();
  }
  LOG(INFO) << "  confidential_space.support_attributes: "
            << (claims.cs_support_attributes.empty()
                    ? "[None Found]"
                    : nlohmann::json(claims.cs_support_attributes).dump());

  LOG(INFO) << "----------------------------";
  return claims;
}

absl::Status AttestationTokenVerifierImpl::EnforcePolicy(
    const ExtractedClaims& claims) const {
  LOG(INFO) << "Enforcing attestation policy...";

  // 1. Hardware Model (Fixed based on config)
  auto status =
      CheckStringMatch(claims.hwmodel, internal_config_.expected_hw_model,
                       kJwtHwModelAttributeName);
  if (!status.ok()) return status;
  LOG(INFO) << "  [PASS] hwmodel matches '"
            << internal_config_.expected_hw_model << "'";

  // 2. Secure Boot: Must be enabled for TEE protection.
  if (!policy_.skip_secboot()) {
    if (!claims.secboot.has_value() || *claims.secboot != true) {
      return absl::PermissionDeniedError(
          "Policy violation: Secure Boot must be enabled.");
    }
    LOG(INFO) << "  [PASS] Secure Boot is enabled.";
  } else {
    LOG(INFO) << "  [INFO] Secure Boot check skipped (explicitly disabled by "
                 "policy).";
  }

  // 3. Debugging Status: Ensures no platform debugging is possible.
  if (!policy_.allow_debug()) {
    // Check hardware debug status (dbgstat)
    status =
        CheckStringMatch(claims.dbgstat, "disabled", kJwtDbgstatAttributeName);
    if (!status.ok()) return status;

    // Check Confidential Space operational attributes for the 'DEBUG' flag.
    for (const auto& attr : claims.cs_support_attributes) {
      if (attr == "DEBUG") {
        return absl::PermissionDeniedError(
            "Policy violation: 'DEBUG' found in support_attributes.");
      }
    }
    LOG(INFO) << "  [PASS] Debugging is disabled (dbgstat & attributes).";
  } else {
    LOG(INFO) << "  [INFO] Debug check skipped (allowed by policy).";
  }

  // 4. Identity Binding: Enforce workload origin (Project ID).
  if (!policy_.expected_project_id().empty()) {
    status = CheckStringMatch(claims.gce_project_id,
                              policy_.expected_project_id(), "Project ID");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] Project ID matches expected.";
  }

  // 4. Identity Binding: Enforce workload origin (Service Account).
  if (!policy_.expected_service_account().empty()) {
    bool found = false;
    // Check if the expected service account is present in the token's list.
    for (const auto& sa : claims.google_service_accounts) {
      if (sa == policy_.expected_service_account()) {
        found = true;
        break;
      }
    }
    if (!found) {
      return absl::PermissionDeniedError(
          absl::StrCat("Policy violation: Expected service account '",
                       policy_.expected_service_account(), "' not found."));
    }
    LOG(INFO) << "  [PASS] Expected service account found.";
  }

  // Helper for date enforcement to avoid repetitive code.
  auto enforce_min_date = [](const std::string& min_date_str,
                             const absl::StatusOr<std::string>& actual_date_str,
                             absl::string_view label) -> absl::Status {
    if (min_date_str.empty()) {
      return absl::OkStatus();
    }
    if (!actual_date_str.ok()) {
      return absl::PermissionDeniedError(absl::StrCat(
          "Policy violation: Failed to get '", label,
          "' for minimum date check: ", actual_date_str.status().ToString()));
    }

    absl::Time min_date, actual_date;
    std::string err;

    if (!absl::ParseTime(absl::RFC3339_full, min_date_str, &min_date, &err)) {
      return absl::InternalError(absl::StrCat(
          "Policy configuration error: Invalid minimum date format for '",
          label, "': ", err));
    }

    if (!absl::ParseTime(absl::RFC3339_full, *actual_date_str, &actual_date,
                         &err)) {
      return absl::PermissionDeniedError(
          absl::StrCat("Policy violation: Failed to parse '", label, "' ('",
                       *actual_date_str, "') as RFC3339 timestamp: ", err));
    }

    if (actual_date < min_date) {
      return absl::PermissionDeniedError(
          absl::StrCat("Policy violation: ", label, " is too old. Got ",
                       absl::FormatTime(actual_date), ", required minimum ",
                       absl::FormatTime(min_date)));
    }
    LOG(INFO) << "  [PASS] " << label << " meets minimum requirement.";
    return absl::OkStatus();
  };

  // 5. TCB Freshness: Enforce up-to-date measurements or minimum dates.
  // GCP Software TCB check.
  if (!policy_.allow_outdated_sw_tcb()) {
    status =
        CheckStringMatch(claims.sw_tcb_status, "UpToDate",
                         "SW TCB Status (" + std::string(kSwTcbStatus) + ")");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] SW TCB is UpToDate.";
  }
  // Enforce SW Minimum Date
  status = enforce_min_date(policy_.min_sw_tcb_date(), claims.sw_tcb_date,
                            "SW TCB Date");
  if (!status.ok()) return status;

  // Intel Hardware TCB check.
  if (!policy_.allow_outdated_hw_tcb()) {
    status =
        CheckStringMatch(claims.hw_tcb_status, "UpToDate",
                         "HW TCB Status (" + std::string(kHwTcbStatus) + ")");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] HW TCB is UpToDate.";
  } else {
    LOG(INFO) << "  [INFO] HW TCB UpToDate check skipped (allowed by policy).";
  }
  // Enforce HW Minimum Date (even if UpToDate check is skipped)
  status = enforce_min_date(policy_.min_hw_tcb_date(), claims.hw_tcb_date,
                            "HW TCB Date");
  if (!status.ok()) return status;

  // 6. Workload Measurement: Final integrity check on container.
  if (!policy_.expected_image_digest().empty()) {
    status = CheckStringMatch(claims.image_digest,
                              policy_.expected_image_digest(), "Image Digest");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] Image digest matches expected.";
  } else {
    LOG(INFO) << "  [INFO] Image digest check skipped by policy.";
  }

  LOG(INFO) << "Attestation policy checks passed.";
  return absl::OkStatus();
}

absl::StatusOr<std::string> AttestationTokenVerifierImpl::ExtractNonce(
    const nlohmann::json& payload_json) const {
  if (!payload_json.contains(kJwtNonceAttributeName)) {
    return absl::InternalError(absl::StrCat("Failed to extract nonce claim ('",
                                            kJwtNonceAttributeName,
                                            "') from JWT: claim not found."));
  }

  const auto& nonce_claim = payload_json[kJwtNonceAttributeName];
  std::string nonce_b64;

  // The nonce can be a single string or a list of strings (as per EAT spec).
  if (nonce_claim.is_string()) {
    nonce_b64 = nonce_claim.get<std::string>();
  } else if (nonce_claim.is_array() && !nonce_claim.empty() &&
             nonce_claim[0].is_string()) {
    nonce_b64 = nonce_claim[0].get<std::string>();
  } else {
    return absl::InternalError(
        absl::StrCat("Failed to extract nonce claim: unexpected JSON type: ",
                     nonce_claim.type_name()));
  }

  std::string decoded_nonce;
  if (!absl::Base64Unescape(nonce_b64, &decoded_nonce)) {
    return absl::InternalError(
        absl::StrCat("Failed to Base64 decode nonce: ", nonce_b64));
  }
  if (decoded_nonce.empty()) {
    return absl::InternalError("Decoded nonce is empty.");
  }

  LOG(INFO) << "Successfully extracted public key from nonce ("
            << decoded_nonce.size() << " bytes).";
  return decoded_nonce;
}

absl::Status AttestationTokenVerifierImpl::CheckStringMatch(
    const absl::StatusOr<std::string>& actual, const std::string& expected,
    absl::string_view claim_label) const {
  if (!actual.ok()) {
    return absl::PermissionDeniedError(
        absl::StrCat("Policy violation: Failed to get required '", claim_label,
                     "' claim: ", actual.status().ToString()));
  }
  if (*actual != expected) {
    return absl::PermissionDeniedError(
        absl::StrCat("Policy violation: ", claim_label, " mismatch. Expected '",
                     expected, "', Got '", *actual, "'"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<AttestationTokenVerifier>>
CreateAttestationTokenVerifier(const AttestationPolicy& attestation_policy,
                               const std::string& jwks_payload, bool dump_jwt) {
  auto jwt_status = crypto::tink::JwtSignatureRegister();
  if (!jwt_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to register Tink JWT Signature primitives: ",
                     jwt_status.ToString()));
  }
  absl::StatusOr<std::unique_ptr<crypto::tink::KeysetHandle>> keyset_handle =
      crypto::tink::JwkSetToPublicKeysetHandle(jwks_payload);
  if (!keyset_handle.ok()) {
    return absl::InternalError(
        absl::StrCat("JwkSetToPublicKeysetHandle failed: ",
                     keyset_handle.status().ToString()));
  }
  absl::StatusOr<std::unique_ptr<crypto::tink::JwtPublicKeyVerify>>
      jwt_verifier_or = (*keyset_handle)
                            ->GetPrimitive<crypto::tink::JwtPublicKeyVerify>(
                                crypto::tink::ConfigGlobalRegistry());
  if (!jwt_verifier_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to get JwtPublicKeyVerify primitive: ",
                     jwt_verifier_or.status().ToString()));
  }
  return std::make_unique<AttestationTokenVerifierImpl>(
      attestation_policy, std::move(*jwt_verifier_or), dump_jwt);
}

extern "C" {
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len) {
  if (context == nullptr || jwt_bytes == nullptr || out_public_key == nullptr ||
      out_public_key_len == nullptr) {
    LOG(ERROR) << "FFI verify_jwt_f called with null pointer.";
    return false;
  }
  AttestationTokenVerifier* verifier =
      static_cast<AttestationTokenVerifier*>(context);
  absl::StatusOr<std::string> public_key_or = verifier->VerifyJwt(
      absl::string_view(reinterpret_cast<const char*>(jwt_bytes), jwt_len));
  if (!public_key_or.ok()) {
    LOG(ERROR) << "C++ JWT verification failed: " << public_key_or.status();
    return false;
  }
  if (public_key_or->size() > *out_public_key_len) {
    LOG(ERROR) << "Public key output buffer is too small.";
    *out_public_key_len = public_key_or->size();
    return false;
  }
  memcpy(out_public_key, public_key_or->data(), public_key_or->size());
  *out_public_key_len = public_key_or->size();
  return true;
}
}  // extern "C"

}  // namespace confidential_federated_compute::gcp
