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

// C FFI header defining the interface between C++ client code and the Rust
// library (`client_session_config.rs`) responsible for creating the Oak
// ClientSession configuration.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_SESSION_CONFIG_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_SESSION_CONFIG_H

#include <cstddef>  // For size_t
#include <cstdint>  // For uint8_t

// Forward declare the Oak SessionConfig opaque type used by the FFI.
// Uses nested namespaces to match the structure expected by Oak's FFI bindings.
namespace oak {
namespace session {
namespace bindings {
class SessionConfig;
}  // namespace bindings
using SessionConfig = bindings::SessionConfig;
}  // namespace session
}  // namespace oak

namespace confidential_federated_compute::gcp {

// Extern "C" block ensures C-style linking for FFI compatibility.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function pointer type for the C++ JWT verification callback.
 *
 * This callback is implemented in C++ (`verifier.cc`) and passed to Rust.
 * Rust calls this function during the handshake to verify the server's JWT
 * and extract the public key nonce.
 *
 * @param context Opaque pointer to the C++ verifier object (`MyVerifier`).
 * @param jwt_bytes Pointer to the JWT data received from the server.
 * @param jwt_len Length of the JWT data.
 * @param out_public_key Output buffer provided by Rust to write the extracted
 * public key bytes.
 * @param out_public_key_len In/out parameter. Input: capacity of
 * `out_public_key`. Output: actual size of the written public key, or the
 * required size if the buffer was too small.
 * @return true if JWT verification (including policy checks) and key extraction
 * succeeded, false otherwise.
 */
typedef bool (*verify_jwt_callback)(void* context, const uint8_t* jwt_bytes,
                                    size_t jwt_len,
                                    // Output parameters for public key
                                    uint8_t* out_public_key,
                                    size_t* out_public_key_len);

/**
 * @brief Creates an Oak ClientSession configuration object.
 *
 * Implemented in Rust (`client_session_config.rs`). This function constructs
 * the necessary Oak SessionConfig, incorporating the provided C++ verifier
 * context and callback function pointer for JWT verification. Session binding
 * verification is handled entirely within Rust.
 *
 * @param verifier_context Opaque pointer to the C++ verifier object
 * (`MyVerifier`), passed back during callback invocations.
 * @param verify_jwt_cb Function pointer to the C++ JWT verification callback.
 * @return A raw pointer to the created SessionConfig object. Ownership is
 * transferred to the caller (typically `oak::session::ClientSession::Create`).
 */
oak::session::SessionConfig* create_client_session_config(
    void* verifier_context, verify_jwt_callback verify_jwt_cb);

#ifdef __cplusplus
}  // extern "C"
#endif

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_CLIENT_SESSION_CONFIG_H
