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

#include <cmath>  // For std::modf
#include <memory>
#include <optional>
#include <string>
#include <type_traits>  // For std::remove_reference_t
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"   // For absl::Minutes
#include "nlohmann/json.hpp"  // For JSON parsing
#include "proto/attestation/verification.pb.h"

// Include libcurl for HTTP requests
#include <curl/curl.h>

// --- Tink Includes ---
#include "tink/config/global_registry.h"
#include "tink/jwt/jwk_set_converter.h"
#include "tink/jwt/jwt_public_key_verify.h"
#include "tink/jwt/jwt_signature_config.h"
#include "tink/jwt/jwt_validator.h"
#include "tink/jwt/verified_jwt.h"
#include "tink/util/statusor.h"

namespace gcp_prototype {
namespace {

// --- Constants ---

// URLs for fetching public keys.
constexpr char kWellKnownFileURI[] =
    "https://confidentialcomputing.googleapis.com/.well-known/"
    "openid-configuration";
constexpr char kJwksUriFieldName[] =
    "jwks_uri";  // Field name in the .well-known config.

// Expected JWT standard claim values.
constexpr char kIssuer[] = "https://confidentialcomputing.googleapis.com";
constexpr char kAudience[] =
    "oak_session_noise_v1";  // Must match server request.

// Confidential Computing specific JWT claim names.
constexpr char kJwtNonceAttributeName[] = "eat_nonce";
constexpr char kJwtHwModelAttributeName[] = "hwmodel";
constexpr char kJwtSecbootAttributeName[] = "secboot";
constexpr char kJwtDbgstatAttributeName[] = "dbgstat";
constexpr char kJwtSubmodsAttributeName[] = "submods";
constexpr char kSubmodsContainerFieldName[] = "container";
constexpr char kContainerImageDigestFieldName[] = "image_digest";
constexpr char kSwVersionAttributeName[] = "swversion";
constexpr char kOemIdAttributeName[] = "oemid";

// Callback function for libcurl to write received data into a string.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

// Performs an HTTP GET request using libcurl.
absl::StatusOr<std::string> CurlGet(const std::string& url) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    return absl::InternalError("Failed to initialize curl.");
  }
  // Use RAII for curl handle cleanup.
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_handle(
      curl, curl_easy_cleanup);
  std::string read_buffer;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // Follow redirects.
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "gcp-prototype-client/1.0");

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  if (res == CURLE_OK) {
    // Only get HTTP code if the request succeeded at the curl level.
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    return absl::InternalError(
        absl::StrCat("curl_easy_perform() failed: ", curl_easy_strerror(res)));
  }

  if (http_code != 200) {
    return absl::InternalError(absl::StrCat("HTTP GET failed with code ",
                                            http_code, ": ", read_buffer));
  }
  return read_buffer;
}

// Helper to extract a potentially nested string claim from a parsed JSON object
// using a vector of keys representing the path.
absl::StatusOr<std::string> GetStringClaimFromPath(
    const nlohmann::json& payload_json,
    const std::vector<absl::string_view>& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("Path cannot be empty.");
  }

  // Use a pointer to traverse the JSON object; avoids exceptions on missing
  // keys.
  const nlohmann::json* current_node = &payload_json;
  for (size_t i = 0; i < path.size(); ++i) {
    absl::string_view key = path[i];
    // Check if the current node is an object before attempting lookup.
    if (!current_node->is_object()) {
      return absl::NotFoundError(absl::StrCat(
          "Path traversal failed: Expected object at step ", i, " for key '",
          key, "', but found ", current_node->type_name()));
    }
    // Find the key in the current object.
    auto it = current_node->find(key);
    if (it == current_node->end()) {
      return absl::NotFoundError(absl::StrCat(
          "Claim path not found: Missing key '", key, "' at step ", i));
    }
    // Move to the next node in the path.
    current_node = &(*it);
  }

  // Ensure the final node is a string.
  if (!current_node->is_string()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Claim path found, but final value is not a string (found type: ",
        current_node->type_name(), ") for key '", path.back(), "'"));
  }

  return current_node->get<std::string>();
}

}  // namespace

// --- MyVerifier Implementation ---

MyVerifier::MyVerifier() {
  // Register Tink primitives needed for JWT signature verification.
  // This should ideally be done once at application startup, but placing it
  // here ensures it happens before the verifier is used.
  // CHECK_OK ensures we fail fast if registration fails.
  auto jwt_status = crypto::tink::JwtSignatureRegister();
  CHECK_OK(jwt_status) << "Failed to register Tink JWT Signature primitives.";
}

void MyVerifier::SetPolicy(const AttestationPolicy& policy) {
  policy_ = policy;
  // Log the policy settings for debugging and clarity.
  LOG(INFO) << "Attestation policy set:"
            << " require_debug_disabled=" << policy_.require_debug_disabled
            << ", require_secboot_enabled=" << policy_.require_secboot_enabled
            << ", expected_hw_model=" << policy_.expected_hw_model
            << ", expected_image_digest="
            << (policy_.expected_image_digest.empty()
                    ? "[Not Checked]"
                    : policy_.expected_image_digest);
}

// Initializes the JWT verifier by fetching public keys from Google's endpoints.
absl::Status MyVerifier::Initialize() {
  LOG(INFO) << "Fetching OpenID configuration from: " << kWellKnownFileURI;
  absl::StatusOr<std::string> well_known_payload = CurlGet(kWellKnownFileURI);
  if (!well_known_payload.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to fetch OpenID config: ",
                     well_known_payload.status().ToString()));
  }

  // Parse the OpenID configuration to find the JWKS URI.
  nlohmann::json well_known_json;
  try {
    well_known_json = nlohmann::json::parse(*well_known_payload);
  } catch (const nlohmann::json::parse_error& e) {
    return absl::InternalError(
        absl::StrCat("Failed to parse OpenID config JSON: ", e.what()));
  }

  if (!well_known_json.contains(kJwksUriFieldName)) {
    return absl::InternalError(
        absl::StrCat("OpenID config missing '", kJwksUriFieldName, "' field."));
  }
  std::string jwks_uri =
      well_known_json.at(kJwksUriFieldName)
          .get<std::string>();  // Use .at() for checked access

  LOG(INFO) << "Fetching JWKS from: " << jwks_uri;
  absl::StatusOr<std::string> jwks_payload = CurlGet(jwks_uri);
  if (!jwks_payload.ok()) {
    return absl::InternalError(absl::StrCat("Failed to fetch JWKS: ",
                                            jwks_payload.status().ToString()));
  }

  // Convert the fetched JWK Set (JSON) into a Tink KeysetHandle.
  absl::StatusOr<std::unique_ptr<crypto::tink::KeysetHandle>> keyset_handle =
      crypto::tink::JwkSetToPublicKeysetHandle(*jwks_payload);
  if (!keyset_handle.ok()) {
    return absl::InternalError(
        absl::StrCat("JwkSetToPublicKeysetHandle failed: ",
                     keyset_handle.status().ToString()));
  }

  // Get the JwtPublicKeyVerify primitive from the keyset handle.
  // This primitive object will be used to verify JWT signatures.
  absl::StatusOr<std::unique_ptr<crypto::tink::JwtPublicKeyVerify>>
      jwt_verifier_or = (*keyset_handle)
                            ->GetPrimitive<crypto::tink::JwtPublicKeyVerify>(
                                crypto::tink::ConfigGlobalRegistry());
  if (!jwt_verifier_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to get JwtPublicKeyVerify primitive: ",
                     jwt_verifier_or.status().ToString()));
  }
  jwt_verifier_ = std::move(*jwt_verifier_or);  // Store the primitive.
  LOG(INFO) << "JWT Verifier initialized successfully.";
  return absl::OkStatus();
}

// Oak AttestationVerifier interface method. Currently a stub.
// Future work (b/452094015): Implement Endorsement Verification, potentially
// using Intel Trust Authority (ITA). This involves parsing the endorsements
// proto and validating certificate chains and measurements against a trusted
// source or policy.
absl::StatusOr<oak::attestation::v1::AttestationResults> MyVerifier::Verify(
    std::chrono::time_point<std::chrono::system_clock> now,
    const ::oak::attestation::v1::Evidence& evidence,
    const ::oak::attestation::v1::Endorsements& endorsements) const {
  LOG(INFO)
      << "C++ MyVerifier::Verify (Oak Interface Method) called (STUBBED).";
  // Basic sanity check (example):
  // if (endorsements.tee_endorsements_size() == 0) {
  //   return absl::InvalidArgumentError("Missing TEE endorsements.");
  // }
  oak::attestation::v1::AttestationResults results;
  results.set_status(oak::attestation::v1::AttestationResults::STATUS_SUCCESS);
  // This encryption_public_key field is deprecated in the Oak proto and should
  // not be relied upon. The actual key is extracted from the JWT nonce later.
  results.set_encryption_public_key("DEPRECATED_FAKE_KEY_FROM_OAK_VERIFY_STUB");
  return results;
}

// Verifies the JWT signature, standard claims, and Confidential Computing
// claims against the configured policy. If successful, extracts and returns the
// public key (as raw bytes) from the nonce claim.
absl::StatusOr<std::string> MyVerifier::VerifyJwt(absl::string_view jwt_bytes) {
  LOG(INFO) << "C++ MyVerifier::VerifyJwt called with token (size "
            << jwt_bytes.size() << ").";

  if (!jwt_verifier_) {
    // Should not happen if Initialize() was called successfully.
    return absl::InternalError("JWT Verifier not initialized.");
  }

  // 1. Verify JWT signature and standard claims using Tink.
  absl::StatusOr<crypto::tink::VerifiedJwt> verified_jwt_or =
      VerifyTokenSignatureAndBasicClaims(jwt_bytes);
  if (!verified_jwt_or.ok()) {
    return verified_jwt_or.status();  // Propagate detailed error.
  }
  crypto::tink::VerifiedJwt& verified_jwt = *verified_jwt_or;

  // 2. Parse the full payload for detailed claim extraction.
  absl::StatusOr<nlohmann::json> payload_json_or = ParsePayload(verified_jwt);
  if (!payload_json_or.ok()) {
    return payload_json_or.status();
  }
  const nlohmann::json& payload_json = *payload_json_or;

  // 3. Extract and log specific Confidential Computing claims.
  absl::StatusOr<ExtractedClaims> claims_or =
      ExtractAndLogClaims(verified_jwt, payload_json);
  if (!claims_or.ok()) {
    // Log the error but don't fail immediately, policy check might not need
    // failed claim.
    LOG(ERROR) << "Failed to extract some claims, proceeding to policy check: "
               << claims_or.status();
  }
  // Use extracted claims even if extraction partially failed. Optionals will be
  // empty.
  const ExtractedClaims& claims = claims_or.value_or(ExtractedClaims{});

  // 4. Enforce the configured attestation policy against the extracted claims.
  absl::Status policy_status = EnforcePolicy(claims);
  if (!policy_status.ok()) {
    return policy_status;  // Return the specific policy violation error.
  }

  // 5. Extract and return the public key nonce.
  return ExtractNonce(verified_jwt);
}

// --- Private Helper Methods ---

// Verifies the JWT signature, type, issuer, audience, and timestamps.
absl::StatusOr<crypto::tink::VerifiedJwt>
MyVerifier::VerifyTokenSignatureAndBasicClaims(
    absl::string_view jwt_bytes) const {
  // Build the validator for standard claims.
  absl::StatusOr<crypto::tink::JwtValidator> validator_or =
      crypto::tink::JwtValidatorBuilder()
          .ExpectTypeHeader("JWT")  // Expect "typ":"JWT" header.
          .ExpectIssuer(kIssuer)
          .ExpectAudience(kAudience)
          .SetClockSkew(
              absl::Minutes(5))  // Allow 5 min clock skew for time checks.
          .Build();
  if (!validator_or.ok()) {
    return absl::InternalError(absl::StrCat("Failed to build JWT validator: ",
                                            validator_or.status().ToString()));
  }
  crypto::tink::JwtValidator validator = *validator_or;

  // Perform verification using the initialized Tink primitive.
  absl::StatusOr<crypto::tink::VerifiedJwt> verified_jwt_or =
      jwt_verifier_->VerifyAndDecode(std::string(jwt_bytes), validator);

  if (!verified_jwt_or.ok()) {
    // Make the error message slightly more informative.
    return absl::InternalError(
        absl::StrCat("JWT signature/basic claim verification failed: ",
                     verified_jwt_or.status().ToString()));
  }
  LOG(INFO) << "JWT signature and basic claims verified successfully.";
  return verified_jwt_or;
}

// Parses the JSON payload from a verified JWT.
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

// Extracts and logs relevant Confidential Computing claims.
// Returns a struct containing the extracted claims as optionals/StatusOr.
// Returns OK even if some claims are missing, logging warnings/errors
// internally.
absl::StatusOr<MyVerifier::ExtractedClaims> MyVerifier::ExtractAndLogClaims(
    const crypto::tink::VerifiedJwt& verified_jwt,
    const nlohmann::json& payload_json) const {
  ExtractedClaims claims;
  LOG(INFO) << "--- Extracted JWT Claims ---";

  // Helper lambda for logging parsed string claims
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
  // Helper lambda for logging boolean claims from Tink
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
  // Helper lambda for logging integer claims from Tink
  auto log_tink_int_claim =
      [&](absl::string_view claim_name) -> std::optional<int64_t> {
    auto claim_or =
        verified_jwt.GetNumberClaim(std::string(claim_name));  // Returns double
    if (claim_or.ok()) {
      double val = *claim_or;
      double intpart;
      if (std::modf(val, &intpart) ==
          0.0) {  // Check if it's effectively an integer
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

  // Extract claims and store them in the struct
  claims.hwmodel =
      GetStringClaimFromPath(payload_json, {kJwtHwModelAttributeName});
  log_parsed_claim(kJwtHwModelAttributeName, claims.hwmodel);

  claims.secboot = log_tink_bool_claim(kJwtSecbootAttributeName);

  claims.dbgstat =
      GetStringClaimFromPath(payload_json, {kJwtDbgstatAttributeName});
  log_parsed_claim(kJwtDbgstatAttributeName, claims.dbgstat);

  // Log swversion directly from parsed JSON
  if (payload_json.contains(kSwVersionAttributeName)) {
    const auto& swversion_node = payload_json.at(kSwVersionAttributeName);
    // Log using dump() which handles arrays, strings, etc.
    LOG(INFO) << "  " << kSwVersionAttributeName << ": "
              << swversion_node.dump();
    // Note: We are not storing swversion in ExtractedClaims struct currently.
    // Future work (b/452094015): Add swversion to ExtractedClaims and policy.
  } else {
    LOG(WARNING) << "  " << kSwVersionAttributeName << ": Claim not found.";
  }

  claims.oemid = log_tink_int_claim(kOemIdAttributeName);

  claims.image_digest = GetStringClaimFromPath(
      payload_json, {kJwtSubmodsAttributeName, kSubmodsContainerFieldName,
                     kContainerImageDigestFieldName});
  log_parsed_claim("image_digest", claims.image_digest);

  LOG(INFO) << "----------------------------";
  return claims;  // Return struct containing StatusOr/optional results.
}

// Enforces the configured policy against the extracted claims.
absl::Status MyVerifier::EnforcePolicy(const ExtractedClaims& claims) const {
  LOG(INFO) << "Enforcing attestation policy...";

  // Check hwmodel
  if (!claims.hwmodel.ok()) {
    return absl::PermissionDeniedError(
        absl::StrCat("Attestation policy violation: Failed to get required '",
                     kJwtHwModelAttributeName,
                     "' claim: ", claims.hwmodel.status().ToString()));
  }
  if (*claims.hwmodel != policy_.expected_hw_model) {
    return absl::PermissionDeniedError(absl::StrCat(
        "Attestation policy violation: Expected hwmodel '",
        policy_.expected_hw_model, "' but got '", *claims.hwmodel, "'"));
  }
  LOG(INFO) << "  [PASS] Hardware model matches expected '"
            << policy_.expected_hw_model << "'.";

  // Check secboot
  if (policy_.require_secboot_enabled) {
    if (!claims.secboot.has_value()) {
      return absl::PermissionDeniedError(absl::StrCat(
          "Attestation policy violation: Required claim '",
          kJwtSecbootAttributeName, "' not found or not boolean."));
    }
    if (*claims.secboot != true) {
      return absl::PermissionDeniedError(
          "Attestation policy violation: Secure Boot must be enabled, but was "
          "reported as disabled.");
    }
    LOG(INFO) << "  [PASS] Secure Boot is enabled as required.";
  } else {
    LOG(INFO)
        << "  [INFO] Secure Boot check skipped by policy. Current status: "
        << (claims.secboot.has_value() ? (*claims.secboot ? "true" : "false")
                                       : "[Not Found]");
  }

  // Check dbgstat
  if (policy_.require_debug_disabled) {
    if (!claims.dbgstat.ok()) {
      return absl::PermissionDeniedError(
          absl::StrCat("Attestation policy violation: Failed to get required '",
                       kJwtDbgstatAttributeName,
                       "' claim: ", claims.dbgstat.status().ToString()));
    }
    if (*claims.dbgstat != "disabled") {
      return absl::PermissionDeniedError(
          absl::StrCat("Attestation policy violation: Debugging must be "
                       "disabled, but dbgstat is '",
                       *claims.dbgstat, "'"));
    }
    LOG(INFO) << "  [PASS] Debugging status is 'disabled' as required.";
  } else {
    LOG(INFO) << "  [INFO] Debugging status check skipped by policy (allowed). "
                 "Current status: '"
              << (claims.dbgstat.ok() ? *claims.dbgstat : "[Not Found/Error]")
              << "'";
  }

  // Check image_digest
  if (!policy_.expected_image_digest.empty()) {
    if (!claims.image_digest.ok()) {
      // Return specific error based on why extraction failed (Not Found vs
      // other).
      return absl::PermissionDeniedError(absl::StrCat(
          "Attestation policy violation: Expected image digest '",
          policy_.expected_image_digest, "' but failed to get claim: ",
          claims.image_digest.status().ToString()));
    }
    if (*claims.image_digest != policy_.expected_image_digest) {
      return absl::PermissionDeniedError(
          absl::StrCat("Attestation policy violation: Expected image digest '",
                       policy_.expected_image_digest, "' but got '",
                       *claims.image_digest, "'"));
    }
    LOG(INFO) << "  [PASS] Image digest matches expected.";
  } else {
    LOG(INFO) << "  [INFO] Image digest check skipped by policy (no expected "
                 "digest provided).";
  }

  // Future work (b/452094015): Add policy check for oemid.
  // if (claims.oemid.has_value() && *claims.oemid != 11129) { // 11129 == Intel
  //   return absl::PermissionDeniedError("OEM ID does not match expected value
  //   for Intel.");
  // }
  // Future work (b/452094015): Add policy check for swversion.

  LOG(INFO) << "Attestation policy checks passed.";
  return absl::OkStatus();
}

// Extracts the nonce (containing the public key) from the JWT.
absl::StatusOr<std::string> MyVerifier::ExtractNonce(
    const crypto::tink::VerifiedJwt& verified_jwt) const {
  absl::StatusOr<std::string> nonce_b64_or =
      verified_jwt.GetStringClaim(kJwtNonceAttributeName);
  if (!nonce_b64_or.ok()) {
    // This is critical, return an internal error if extraction fails.
    return absl::InternalError(
        absl::StrCat("Failed to extract nonce claim ('", kJwtNonceAttributeName,
                     "') from JWT: ", nonce_b64_or.status().ToString()));
  }

  std::string decoded_nonce;
  if (!absl::Base64Unescape(*nonce_b64_or, &decoded_nonce)) {
    return absl::InternalError(
        absl::StrCat("Failed to Base64 decode nonce: ", *nonce_b64_or));
  }

  if (decoded_nonce.empty()) {
    return absl::InternalError("Decoded nonce is empty.");
  }

  LOG(INFO) << "Successfully extracted public key from nonce ("
            << decoded_nonce.size() << " bytes).";

  return decoded_nonce;
}

}  // namespace gcp_prototype

// --- Implementation of the C-style FFI wrapper function ---
// This function acts as the bridge between the Rust verifier callback and the
// C++ MyVerifier implementation.
extern "C" {
bool verify_jwt_f(void* context, const uint8_t* jwt_bytes, size_t jwt_len,
                  uint8_t* out_public_key, size_t* out_public_key_len) {
  // Basic null pointer checks for safety.
  if (context == nullptr || jwt_bytes == nullptr || out_public_key == nullptr ||
      out_public_key_len == nullptr) {
    LOG(ERROR) << "FFI verify_jwt_f called with null pointer.";
    return false;
  }

  // Cast the opaque context pointer back to our C++ verifier object.
  gcp_prototype::MyVerifier* verifier =
      static_cast<gcp_prototype::MyVerifier*>(context);

  // Call the main verification logic.
  absl::StatusOr<std::string> public_key_or = verifier->VerifyJwt(
      absl::string_view(reinterpret_cast<const char*>(jwt_bytes), jwt_len));

  // Check if verification succeeded. If not, log the specific error.
  if (!public_key_or.ok()) {
    LOG(ERROR) << "C++ JWT verification failed: " << public_key_or.status();
    return false;  // Indicate failure to Rust.
  }

  // Check if the output buffer provided by Rust is large enough.
  if (public_key_or->size() > *out_public_key_len) {
    LOG(ERROR) << "Public key output buffer is too small. Need "
               << public_key_or->size() << " bytes, have "
               << *out_public_key_len;
    // Report the required size back to Rust.
    *out_public_key_len = public_key_or->size();
    return false;  // Indicate failure due to buffer size.
  }

  // Copy the extracted public key (nonce) into the Rust buffer.
  memcpy(out_public_key, public_key_or->data(), public_key_or->size());
  // Update the length parameter to reflect the actual size copied.
  *out_public_key_len = public_key_or->size();

  // Indicate success to Rust.
  return true;
}
}  // extern "C"
