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

#include "verifier.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "http_client.h"
#include "json_util.h"
#include "nlohmann/json.hpp"
#include "proto/attestation/verification.pb.h"
#include "tink/config/global_registry.h"
#include "tink/jwt/jwk_set_converter.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/jwt_signature_config.h"
#include "tink/jwt/jwt_validator.h"
#include "tink/jwt/verified_jwt.h"
#include "tink/util/statusor.h"

namespace gcp_prototype {
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

}  // namespace

MyVerifier::MyVerifier() {
  auto jwt_status = crypto::tink::JwtSignatureRegister();
  CHECK_OK(jwt_status) << "Failed to register Tink JWT Signature primitives.";
}

void MyVerifier::SkipPolicyEnforcement(bool skip) {
  skip_policy_enforcement_ = skip;
}

void MyVerifier::SetDumpJwt(bool dump) { dump_jwt_ = dump; }

void MyVerifier::SetPolicy(const AttestationPolicy& policy) {
  client_policy_ = policy;
  std::string verifier_type_str;

  // Set the internal configuration based on the requested verifier type.
  switch (client_policy_.verifier_type) {
    case AttestationPolicy::VerifierType::kGca:
      internal_config_ = {
          .expected_issuer = "https://confidentialcomputing.googleapis.com",
          .jwks_url =
              "https://www.googleapis.com/service_accounts/v1/metadata/jwk/"
              "signer@confidentialspace-sign.iam.gserviceaccount.com",
          .expected_hw_model = "GCP_INTEL_TDX"};
      verifier_type_str = "GCA (Legacy)";
      break;
    case AttestationPolicy::VerifierType::kIta:
      // ITA is the configured Root of Trust, relying on their JWKS endpoint.
      internal_config_ = {
          .expected_issuer = "https://portal.trustauthority.intel.com",
          .jwks_url = "https://portal.trustauthority.intel.com/certs",
          .expected_hw_model = "INTEL_TDX"};
      verifier_type_str = "ITA (Intel RoT)";
      break;
  }

  // Log the effective policy for debugging and verification purposes.
  LOG(INFO)
      << "Attestation policy set:"
      << " verifier_type=" << verifier_type_str
      << ", require_debug_disabled=" << client_policy_.require_debug_disabled
      << ", require_secboot_enabled=" << client_policy_.require_secboot_enabled
      << ", require_sw_tcb_uptodate=" << client_policy_.require_sw_tcb_uptodate
      << ", require_hw_tcb_uptodate=" << client_policy_.require_hw_tcb_uptodate
      << ", min_sw_tcb_date="
      << (client_policy_.min_sw_tcb_date.has_value()
              ? absl::FormatTime(*client_policy_.min_sw_tcb_date)
              : "[Not Checked]")
      << ", min_hw_tcb_date="
      << (client_policy_.min_hw_tcb_date.has_value()
              ? absl::FormatTime(*client_policy_.min_hw_tcb_date)
              : "[Not Checked]")
      << ", expected_project_id="
      << (client_policy_.expected_project_id.empty()
              ? "[Not Checked]"
              : client_policy_.expected_project_id)
      << ", expected_service_account="
      << (client_policy_.expected_service_account.empty()
              ? "[Not Checked]"
              : client_policy_.expected_service_account)
      << ", expected_image_digest="
      << (client_policy_.expected_image_digest.empty()
              ? "[Not Checked]"
              : client_policy_.expected_image_digest);
  LOG(INFO) << "Internal verifier config set:"
            << " expected_issuer=" << internal_config_.expected_issuer
            << ", jwks_url=" << internal_config_.jwks_url
            << ", expected_hw_model=" << internal_config_.expected_hw_model;
}

absl::Status MyVerifier::Initialize() {
  if (internal_config_.jwks_url.empty()) {
    return absl::FailedPreconditionError(
        "Verifier policy not set. SetPolicy() must be called before "
        "Initialize().");
  }

  // Fetch the public key set (JWKS) from the attestation provider's endpoint.
  LOG(INFO) << "Fetching JWKS from: " << internal_config_.jwks_url;
  absl::StatusOr<std::string> jwks_payload =
      gcp_prototype::CurlGet(internal_config_.jwks_url);
  if (!jwks_payload.ok()) {
    return absl::InternalError(absl::StrCat("Failed to fetch JWKS: ",
                                            jwks_payload.status().ToString()));
  }

  // Convert the JWKS into a Tink keyset handle for signature verification.
  absl::StatusOr<std::unique_ptr<crypto::tink::KeysetHandle>> keyset_handle =
      crypto::tink::JwkSetToPublicKeysetHandle(*jwks_payload);
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
  jwt_verifier_ = std::move(*jwt_verifier_or);
  LOG(INFO) << "JWT Verifier initialized successfully.";
  return absl::OkStatus();
}

absl::StatusOr<oak::attestation::v1::AttestationResults> MyVerifier::Verify(
    std::chrono::time_point<std::chrono::system_clock> now,
    const ::oak::attestation::v1::Evidence& evidence,
    const ::oak::attestation::v1::Endorsements& endorsements) const {
  LOG(INFO)
      << "C++ MyVerifier::Verify (Oak Interface Method) called (STUBBED).";
  // This is a stub for the full Oak AttestationVerifier interface, as
  // verification is done directly via VerifyJwt.
  oak::attestation::v1::AttestationResults results;
  results.set_status(oak::attestation::v1::AttestationResults::STATUS_SUCCESS);
  results.set_encryption_public_key("DEPRECATED_FAKE_KEY_FROM_OAK_VERIFY_STUB");
  return results;
}

absl::StatusOr<std::string> MyVerifier::VerifyJwt(absl::string_view jwt_bytes) {
  LOG(INFO) << "C++ MyVerifier::VerifyJwt called with token (size "
            << jwt_bytes.size() << ").";
  if (!jwt_verifier_) {
    return absl::InternalError("JWT Verifier not initialized.");
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
MyVerifier::VerifyTokenSignatureAndBasicClaims(
    absl::string_view jwt_bytes) const {
  if (internal_config_.expected_issuer.empty()) {
    return absl::InternalError(
        "Internal verifier config error: expected_issuer is empty.");
  }

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
  crypto::tink::JwtValidator validator = *validator_or;

  // Verify signature using JWKS and decode the token while validating claims.
  absl::StatusOr<crypto::tink::VerifiedJwt> verified_jwt_or =
      jwt_verifier_->VerifyAndDecode(std::string(jwt_bytes), validator);
  if (!verified_jwt_or.ok()) {
    return absl::InternalError(
        absl::StrCat("JWT signature/basic claim verification failed: ",
                     verified_jwt_or.status().ToString()));
  }
  LOG(INFO) << "JWT signature and basic claims verified successfully.";
  return verified_jwt_or;
}

absl::StatusOr<nlohmann::json> MyVerifier::ParsePayload(
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

absl::StatusOr<MyVerifier::ExtractedClaims> MyVerifier::ExtractAndLogClaims(
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
  claims.hwmodel = gcp_prototype::GetStringClaimFromPath(
      payload_json, {kJwtHwModelAttributeName});
  log_parsed_claim(kJwtHwModelAttributeName, claims.hwmodel);

  claims.secboot = log_tink_bool_claim(kJwtSecbootAttributeName);

  claims.dbgstat = gcp_prototype::GetStringClaimFromPath(
      payload_json, {kJwtDbgstatAttributeName});
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
  claims.image_digest = gcp_prototype::GetStringClaimFromPath(
      payload_json, {kJwtSubmodsAttributeName, kSubmodsContainerFieldName,
                     kContainerImageDigestFieldName});
  log_parsed_claim("image_digest", claims.image_digest);

  // 3. GCP Identity Claims
  auto sa_list_or = gcp_prototype::GetStringListClaimFromPath(
      payload_json, {kGoogleServiceAccounts});
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
  claims.gce_project_id = gcp_prototype::GetStringClaimFromPath(
      payload_json, {kJwtSubmodsAttributeName, kSubmodGce, kGceProjectId});
  log_parsed_claim("gce.project_id", claims.gce_project_id);

  // 4. TCB Statuses (Delegated Intel Verification)
  // GCP Software TCB Status (kernel, CS agent, etc.).
  claims.sw_tcb_status =
      gcp_prototype::GetStringClaimFromPath(payload_json, {kTdx, kSwTcbStatus});
  log_parsed_claim("tdx.gcp_attester_tcb_status (SW)", claims.sw_tcb_status);

  claims.sw_tcb_date =
      gcp_prototype::GetStringClaimFromPath(payload_json, {kTdx, kSwTcbDate});
  log_parsed_claim("tdx.gcp_attester_tcb_date (SW)", claims.sw_tcb_date);

  // Intel Hardware TCB Status (microcode, TDX module).
  claims.hw_tcb_status =
      gcp_prototype::GetStringClaimFromPath(payload_json, {kTdx, kHwTcbStatus});
  log_parsed_claim("tdx.attester_tcb_status (HW)", claims.hw_tcb_status);

  claims.hw_tcb_date =
      gcp_prototype::GetStringClaimFromPath(payload_json, {kTdx, kHwTcbDate});
  log_parsed_claim("tdx.attester_tcb_date (HW)", claims.hw_tcb_date);

  // 5. Confidential Space Support Attributes (e.g., DEBUG flag)
  auto support_attrs_or = gcp_prototype::GetStringListClaimFromPath(
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

absl::Status MyVerifier::EnforcePolicy(const ExtractedClaims& claims) const {
  if (skip_policy_enforcement_) {
    LOG(WARNING) << "Policy enforcement is being SKIPPED due to debug flag.";
    return absl::OkStatus();
  }
  LOG(INFO) << "Enforcing attestation policy...";

  // 1. Hardware Model: Must match the expected TEE type (e.g., INTEL_TDX).
  if (internal_config_.expected_hw_model.empty()) {
    return absl::InternalError("config error: expected_hw_model is empty.");
  }
  absl::Status status =
      CheckStringMatch(claims.hwmodel, internal_config_.expected_hw_model,
                       kJwtHwModelAttributeName);
  if (!status.ok()) return status;
  LOG(INFO) << "  [PASS] hwmodel matches '"
            << internal_config_.expected_hw_model << "'";

  // 2. Secure Boot: Must be enabled for TEE protection.
  if (client_policy_.require_secboot_enabled) {
    if (!claims.secboot.has_value() || *claims.secboot != true) {
      return absl::PermissionDeniedError(
          "Policy violation: Secure Boot must be enabled.");
    }
    LOG(INFO) << "  [PASS] Secure Boot is enabled.";
  }

  // 3. Debugging Status: Ensures no platform debugging is possible.
  if (client_policy_.require_debug_disabled) {
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
  if (!client_policy_.expected_project_id.empty()) {
    status = CheckStringMatch(claims.gce_project_id,
                              client_policy_.expected_project_id, "Project ID");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] Project ID matches expected.";
  }

  // 4. Identity Binding: Enforce workload origin (Service Account).
  if (!client_policy_.expected_service_account.empty()) {
    bool found = false;
    // Check if the expected service account is present in the token's list.
    for (const auto& sa : claims.google_service_accounts) {
      if (sa == client_policy_.expected_service_account) {
        found = true;
        break;
      }
    }
    if (!found) {
      return absl::PermissionDeniedError(absl::StrCat(
          "Policy violation: Expected service account '",
          client_policy_.expected_service_account, "' not found."));
    }
    LOG(INFO) << "  [PASS] Expected service account found.";
  }

  // 5. TCB Freshness: Enforce up-to-date measurements or minimum dates.

  // Helper for date enforcement to avoid repetitive code.
  auto enforce_min_date = [](const std::optional<absl::Time>& min_date,
                             const absl::StatusOr<std::string>& actual_date_str,
                             absl::string_view label) -> absl::Status {
    if (!min_date.has_value()) {
      return absl::OkStatus();
    }
    if (!actual_date_str.ok()) {
      return absl::PermissionDeniedError(absl::StrCat(
          "Policy violation: Failed to get '", label,
          "' for minimum date check: ", actual_date_str.status().ToString()));
    }
    absl::Time actual_date;
    std::string err;
    // Parse RFC 3339 timestamp (e.g., "2025-08-14T00:00:00Z")
    if (!absl::ParseTime(absl::RFC3339_full, *actual_date_str, &actual_date,
                         &err)) {
      return absl::PermissionDeniedError(
          absl::StrCat("Policy violation: Failed to parse '", label, "' ('",
                       *actual_date_str, "') as RFC3339 timestamp: ", err));
    }
    if (actual_date < *min_date) {
      return absl::PermissionDeniedError(
          absl::StrCat("Policy violation: ", label, " is too old. Got ",
                       absl::FormatTime(actual_date), ", required minimum ",
                       absl::FormatTime(*min_date)));
    }
    LOG(INFO) << "  [PASS] " << label << " meets minimum requirement.";
    return absl::OkStatus();
  };

  // GCP Software TCB check.
  if (client_policy_.require_sw_tcb_uptodate) {
    status =
        CheckStringMatch(claims.sw_tcb_status, "UpToDate",
                         "SW TCB Status (" + std::string(kSwTcbStatus) + ")");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] SW TCB is UpToDate.";
  }
  // Enforce SW Minimum Date
  status = enforce_min_date(client_policy_.min_sw_tcb_date, claims.sw_tcb_date,
                            "SW TCB Date");
  if (!status.ok()) return status;

  // Intel Hardware TCB check.
  if (client_policy_.require_hw_tcb_uptodate) {
    status =
        CheckStringMatch(claims.hw_tcb_status, "UpToDate",
                         "HW TCB Status (" + std::string(kHwTcbStatus) + ")");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] HW TCB is UpToDate.";
  } else {
    LOG(INFO) << "  [INFO] HW TCB UpToDate check skipped (allowed by policy).";
  }
  // Enforce HW Minimum Date (even if UpToDate check is skipped)
  status = enforce_min_date(client_policy_.min_hw_tcb_date, claims.hw_tcb_date,
                            "HW TCB Date");
  if (!status.ok()) return status;

  // 6. Workload Measurement: Final integrity check on container.
  if (!client_policy_.expected_image_digest.empty()) {
    status =
        CheckStringMatch(claims.image_digest,
                         client_policy_.expected_image_digest, "Image Digest");
    if (!status.ok()) return status;
    LOG(INFO) << "  [PASS] Image digest matches expected.";
  } else {
    LOG(INFO) << "  [INFO] Image digest check skipped by policy.";
  }

  LOG(INFO) << "Attestation policy checks passed.";
  return absl::OkStatus();
}

absl::StatusOr<std::string> MyVerifier::ExtractNonce(
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

absl::Status MyVerifier::CheckStringMatch(
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

}  // namespace gcp_prototype

extern "C" {
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len) {
  if (context == nullptr || jwt_bytes == nullptr || out_public_key == nullptr ||
      out_public_key_len == nullptr) {
    LOG(ERROR) << "FFI verify_jwt_f called with null pointer.";
    return false;
  }
  gcp_prototype::MyVerifier* verifier =
      static_cast<gcp_prototype::MyVerifier*>(context);
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
