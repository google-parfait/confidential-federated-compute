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

// Implementation of the helper function to fetch attestation tokens from the
// GCP Confidential Space agent using libcurl.

#include "attestation.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"  // For formatting the JSON request.
// Include libcurl for HTTP requests via Unix socket.
#include <curl/curl.h>
// Include nlohmann/json for potential future parsing needs (though not used
// currently).
#include <nlohmann/json.hpp>

namespace gcp_prototype {
namespace {

// Constants for interacting with the Confidential Space agent.
constexpr char kLocalhostTokenUrl[] = "http://localhost/v1/token";
constexpr char kLauncherSocketPath[] = "/run/container_launcher/teeserver.sock";
// Template for the JSON request body sent to the agent.
constexpr char kAttestationJsonRequestTemplate[] = R"json(
{
  "audience": "$0",
  "token_type": "OIDC",
  "nonces": [
    "$1"
  ]
})json";
// Audience expected by the client's verifier.
constexpr char kAudience[] = "oak_session_noise_v1";

// Callback function for libcurl to write received HTTP response body into a
// string.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            void* userp) {
  // Append data to the std::string pointed to by userp.
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;  // Return total bytes handled.
}

// Helper function to perform an HTTP POST request with a JSON payload
// over a Unix domain socket using libcurl.
absl::StatusOr<std::string> PostJsonViaUnixSocket(
    const std::string& url, const std::string& socket_path,
    const std::string& json_payload) {
  // Initialize curl easy handle.
  CURL* curl = curl_easy_init();
  if (!curl) {
    return absl::InternalError("Failed to initialize curl.");
  }
  // Use RAII (unique_ptr with custom deleter) for curl handle cleanup.
  std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_handle(
      curl, curl_easy_cleanup);

  std::string read_buffer;               // String to store the response body.
  struct curl_slist* headers = nullptr;  // List for custom HTTP headers.
  // Use RAII for header list cleanup.
  std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> header_list_guard(
      headers, curl_slist_free_all);

  // Set required headers.
  headers = curl_slist_append(headers, "Content-Type: application/json");
  // Metadata-Flavor header is required by GCP metadata/attestation services.
  headers = curl_slist_append(headers, "Metadata-Flavor: Google");
  // Update the pointer in the RAII guard in case append reallocated the list.
  header_list_guard.reset(headers);

  // Configure curl options:
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());  // Target URL (localhost).
  curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH,
                   socket_path.c_str());  // Unix socket path.
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER,
                   header_list_guard.get());  // Custom headers.
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS,
                   json_payload.c_str());  // Request body.
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                   WriteCallback);  // Response body callback.
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,
                   &read_buffer);  // String for response body.
  // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // Uncomment for curl debug
  // logging.

  // Perform the HTTP request.
  CURLcode res = curl_easy_perform(curl);

  long http_code = 0;
  if (res == CURLE_OK) {
    // Only get the response code if the transfer succeeded.
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  } else {
    // Curl error (e.g., connection failed).
    return absl::InternalError(
        absl::StrCat("curl_easy_perform() failed: ", curl_easy_strerror(res)));
  }

  // Check the HTTP status code from the agent.
  if (http_code != 200) {
    // Agent returned an HTTP error (e.g., bad request, internal server error).
    return absl::InternalError(
        absl::StrCat("Attestation agent returned HTTP status ", http_code, ": ",
                     read_buffer));
  }

  // Success: return the response body (the token).
  return read_buffer;
}

}  // namespace

// Fetches the attestation token using the helper function.
absl::StatusOr<std::string> GetAttestationToken(absl::string_view nonce) {
  LOG(INFO) << "Fetching real attestation token via libcurl for nonce (len "
            << nonce.length() << ")...";

  // Basic validation on nonce length (based on typical Base64 P-256 key size).
  if (nonce.length() < 8 || nonce.length() > 88) {
    // Check based on agent's known limits (can prevent unnecessary HTTP calls).
    return absl::InvalidArgumentError(
        absl::StrCat("Nonce length (", nonce.length(),
                     ") is outside the likely required range [8, 88]."));
  }

  // Construct the JSON request payload using the provided audience and nonce.
  std::string json_request =
      absl::Substitute(kAttestationJsonRequestTemplate, kAudience, nonce);

  // Make the HTTP POST request via Unix socket using the libcurl helper.
  LOG(INFO) << "Sending attestation request: " << json_request;
  absl::StatusOr<std::string> response_or = PostJsonViaUnixSocket(
      kLocalhostTokenUrl, kLauncherSocketPath, json_request);

  if (!response_or.ok()) {
    // Propagate errors from the HTTP request.
    return response_or.status();
  }
  std::string response_body = *response_or;
  LOG(INFO) << "Received attestation response (token size "
            << response_body.length() << ").";
  // Log raw response only at VLOG(1) or higher for potentially sensitive
  // tokens.
  VLOG(1) << "Raw response body (token): <<<" << response_body << ">>>";

  // The response body *is* the JWT token string. No JSON parsing needed here.
  return response_body;
}

}  // namespace gcp_prototype
