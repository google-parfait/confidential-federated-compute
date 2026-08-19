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

#include "attestation_token_provider.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "http_client.h"

namespace confidential_federated_compute::gcp {
namespace {

// Minimum interval between consecutive ITA token requests from this process.
constexpr absl::Duration kMinAttestationInterval = absl::Milliseconds(2000);

// Internal configuration structure for specifying how to request a token
// from the local Confidential Space agent.
struct AttestationTokenProviderConfig {
  // The full URL path to request the token from (e.g., v1/token vs
  // v1/intel/token).
  std::string token_url;
  // The 'token_type' parameter value expected by the agent for this provider.
  std::string token_type;
};

// Configuration for standard Google Cloud Attestation (GCA).
const AttestationTokenProviderConfig kGcaConfig = {
    .token_url = "http://localhost/v1/token",
    .token_type = "OIDC",
};

// Configuration for Intel Trust Authority (ITA).
// ITA tokens are requested via a specific endpoint and use PRINCIPAL_TAGS
// to trigger remote verification.
const AttestationTokenProviderConfig kItaConfig = {
    .token_url = "http://localhost/v1/intel/token",
    .token_type = "PRINCIPAL_TAGS",
};

// Template for the JSON request body sent to the local Confidential Space
// agent. $0: audience, $1: token_type, $2: nonce
constexpr char kAttestationJsonRequestTemplate[] = R"json(
{
  "audience": "$0",
  "token_type": "$1",
  "nonces": [
    "$2"
  ]
})json";

// The audience claim to be included in the token.
constexpr char kAudience[] = "oak_session_noise_v1";

/**
 * @brief Concrete implementation of AttestationTokenProvider that communicates
 * with the local Confidential Space agent via a Unix domain socket.
 *
 * Includes rate limiting: concurrent calls are serialized via a mutex, and
 * a minimum interval is enforced between consecutive ITA requests to avoid
 * hitting Intel Trust Authority's rate limit (2 QPS).
 */
class AttestationTokenProviderImpl : public AttestationTokenProvider {
 public:
  explicit AttestationTokenProviderImpl(AttestationTokenProviderConfig config)
      : config_(std::move(config)) {}

  absl::StatusOr<std::string> GetAttestationToken(
      absl::string_view nonce) override {
    // Serialize concurrent attestation requests and enforce minimum interval
    // between consecutive calls to avoid ITA rate limiting (HTTP 429).
    absl::MutexLock lock(&mu_);
    absl::Time now = absl::Now();
    absl::Duration since_last = now - last_call_time_;
    if (since_last < kMinAttestationInterval) {
      absl::Duration wait = kMinAttestationInterval - since_last;
      LOG(INFO) << "AttestationTokenProviderImpl: Rate limiting — waiting "
                << absl::ToInt64Milliseconds(wait)
                << "ms before next ITA call.";
      absl::SleepFor(wait);
    }

    LOG(INFO) << "AttestationTokenProviderImpl: Fetching token for "
              << config_.token_type << " provider (nonce len " << nonce.length()
              << ")...";

    // Basic validation to ensure the nonce is reasonable before sending.
    if (nonce.length() < 8 || nonce.length() > 256) {
      return absl::InvalidArgumentError(
          absl::StrCat("Nonce length (", nonce.length(),
                       ") is outside the expected range [8, 256]."));
    }

    // Construct the JSON request payload.
    std::string json_request = absl::Substitute(
        kAttestationJsonRequestTemplate, kAudience, config_.token_type, nonce);

    LOG(INFO) << "AttestationTokenProviderImpl: Sending request to "
              << config_.token_url;

    // The Confidential Space agent listens on this fixed Unix socket path.
    constexpr char kLauncherSocketPath[] =
        "/run/container_launcher/teeserver.sock";

    // Perform the HTTP POST request via the Unix socket.
    absl::StatusOr<std::string> response_or = PostJsonViaUnixSocket(
        config_.token_url, kLauncherSocketPath, json_request);

    // Record the time of this call regardless of success/failure,
    // so we rate-limit even after errors.
    last_call_time_ = absl::Now();

    if (!response_or.ok()) {
      LOG(ERROR) << "AttestationTokenProviderImpl: Failed to fetch token: "
                 << response_or.status();
      return response_or.status();
    }

    LOG(INFO) << "AttestationTokenProviderImpl: Received token (size "
              << response_or->length() << ").";
    return *response_or;
  }

 private:
  AttestationTokenProviderConfig config_;
  absl::Mutex mu_;
  absl::Time last_call_time_ ABSL_GUARDED_BY(mu_) = absl::InfinitePast();
};

}  // namespace

// Factory implementation to create the appropriate provider.
std::unique_ptr<AttestationTokenProvider> CreateAttestationTokenProvider(
    ProviderType type) {
  switch (type) {
    case ProviderType::kGca:
      LOG(INFO) << "Creating GCA Attestation Token Provider.";
      return std::make_unique<AttestationTokenProviderImpl>(kGcaConfig);
    case ProviderType::kIta:
      LOG(INFO) << "Creating ITA Attestation Token Provider.";
      return std::make_unique<AttestationTokenProviderImpl>(kItaConfig);
    default:
      LOG(FATAL) << "Unknown ProviderType specified in factory.";
      return nullptr;  // Unreachable
  }
}

}  // namespace confidential_federated_compute::gcp
