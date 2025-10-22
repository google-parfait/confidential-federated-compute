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

//! Rust FFI module for creating an Oak ClientSession configuration.
//!
//! This module provides the `create_client_session_config` C FFI function,
//! which allows C++ code to initialize an Oak ClientSession. It uses a C++
//! callback (`verify_jwt_callback`) for JWT verification and public key
//! extraction, while handling the session binding verification directly in Rust
//! using the `p256` and `ecdsa` crates.

use anyhow::Result;
use oak_proto_rust::oak::{attestation::v1::Assertion, session::v1::SessionBinding};
use oak_session::{
    aggregators::PassThrough, // Aggregator for simple attestation evidence passthrough.
    attestation::AttestationType,
    config::SessionConfig,
    handshake::HandshakeType,
    verifier::{BoundAssertionVerificationError, BoundAssertionVerifier, VerifiedBoundAssertion},
};
// Import crypto crates for P-256 ECDSA verification.
use ecdsa::{signature::Verifier as _, Signature}; /* Import the Verifier trait extension
                                                   * methods. */
use p256::{ecdsa::VerifyingKey, EncodedPoint};
use std::{ffi::c_void, sync::Arc};

// --- FFI Type Definitions ---

/// C function pointer type matching `verify_jwt_callback` in the C++ header.
/// This function is implemented in C++ and called by Rust.
type VerifyJwtCallback = extern "C" fn(
    context: *mut c_void,           // Opaque C++ verifier object pointer.
    jwt_bytes: *const u8,           // Raw JWT bytes from the server.
    jwt_len: usize,                 // Length of JWT bytes.
    out_public_key: *mut u8,        // Output buffer for the extracted public key.
    out_public_key_len: *mut usize, // In/out: buffer capacity / actual key size.
) -> bool; // Returns true on successful verification and extraction.

/// Wrapper struct to make the C++ context pointer `Send + Sync`.
/// This is `unsafe` because Rust cannot guarantee the thread-safety of the C++
/// object. The C++ side must ensure thread safety if the verifier is used
/// across multiple threads (which is unlikely in this specific client scenario
/// but important for the FFI contract).
#[derive(Clone, Copy, Debug)]
struct SendSyncRawPtr(*mut c_void);

unsafe impl Send for SendSyncRawPtr {}
unsafe impl Sync for SendSyncRawPtr {}

// --- Oak Verifier Trait Implementations ---

/// Represents a successfully verified assertion (JWT) and holds the data needed
/// for the subsequent binding verification step.
#[derive(Debug)]
struct FfiVerifiedAssertion {
    /// The original assertion proto (mainly for Oak Session internals).
    assertion: Assertion,
    /// The opaque C++ verifier context pointer (kept for consistency, though
    /// not used in verify_binding).
    #[allow(dead_code)] // Avoid warning for unused field.
    verifier_context: SendSyncRawPtr,
    /// The public key extracted from the JWT's `eat_nonce` claim by the C++
    /// callback.
    public_key: Vec<u8>,
}

impl VerifiedBoundAssertion for FfiVerifiedAssertion {
    /// Returns a reference to the original assertion proto.
    fn assertion(&self) -> &Assertion {
        &self.assertion
    }

    /// Verifies the session binding signature provided by the server.
    /// This implementation performs the ECDSA-P256 signature verification in
    /// Rust.
    ///
    /// # Arguments
    ///
    /// * `bound_data` - The data that was signed (the Noise session ID).
    /// * `binding` - The `SessionBinding` proto containing the signature bytes.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the signature is valid.
    /// * `Err(BoundAssertionVerificationError)` - If parsing or verification
    ///   fails.
    fn verify_binding(
        &self,
        bound_data: &[u8], // This is the session_id from the Noise handshake.
        binding: &SessionBinding,
    ) -> Result<(), BoundAssertionVerificationError> {
        eprintln!("---> Rust: FfiVerifiedAssertion::verify_binding called (Rust crypto).");

        // 1. Parse the raw public key bytes (expected SEC1 uncompressed format). The
        //    key bytes were extracted from the JWT nonce by the C++ callback.
        let encoded_point = EncodedPoint::from_bytes(&self.public_key).map_err(|e| {
            // Map crypto errors to the specific Oak verifier error type.
            BoundAssertionVerificationError::GenericFailure {
                error_msg: format!("Failed to parse public key bytes from nonce: {}", e),
            }
        })?;

        let verifying_key = VerifyingKey::from_encoded_point(&encoded_point).map_err(|e| {
            BoundAssertionVerificationError::GenericFailure {
                error_msg: format!("Failed to create verifying key from nonce public key: {}", e),
            }
        })?;

        // 2. Parse the raw signature bytes received from the server. Use `from_slice`
        //    as the binding field is a Vec<u8>, not a fixed array.
        let signature = Signature::from_slice(&binding.binding).map_err(|e| {
            BoundAssertionVerificationError::BindingVerificationFailure {
                error_msg: format!("Failed to parse server signature bytes: {}", e),
            }
        })?;

        // 3. Verify the signature against the session ID using the public key. The
        //    `verify` method from the `ecdsa::signature::Verifier` trait handles the
        //    necessary hashing internally (matching the server's `Signer` use).
        verifying_key.verify(bound_data, &signature).map_err(|e| {
            BoundAssertionVerificationError::BindingVerificationFailure {
                error_msg: format!("Session binding signature verification failed: {}", e),
            }
        })?;

        eprintln!("---> Rust: Session binding verified successfully in Rust.");
        Ok(())
    }
}

/// Implements the Oak `BoundAssertionVerifier` trait using the C++ FFI
/// callback.
struct FfiVerifier {
    /// Opaque pointer to the C++ `MyVerifier` object.
    verifier_context: SendSyncRawPtr,
    /// Function pointer to the C++ `verify_jwt_f` function.
    verify_jwt_cb: VerifyJwtCallback,
}

impl BoundAssertionVerifier for FfiVerifier {
    /// Called by Oak Session during the handshake to verify the server's
    /// assertion (JWT). This function calls the C++ `verify_jwt_f`
    /// callback. If successful, it creates and returns an
    /// `FfiVerifiedAssertion` containing the extracted public key.
    ///
    /// # Arguments
    ///
    /// * `assertion` - The `Assertion` proto containing the raw JWT bytes in
    ///   its `content` field.
    ///
    /// # Returns
    ///
    /// * `Ok(Box<dyn VerifiedBoundAssertion>)` - On successful JWT verification
    ///   by C++.
    /// * `Err(BoundAssertionVerificationError)` - If the C++ callback returns
    ///   failure.
    fn verify_assertion(
        &self,
        assertion: &Assertion,
    ) -> Result<Box<dyn VerifiedBoundAssertion>, BoundAssertionVerificationError> {
        eprintln!("---> Rust: FfiVerifier::verify_assertion called. Calling C++...");

        // Allocate a buffer for the public key. P-256 uncompressed is 65 bytes.
        // Provide a slightly larger buffer just in case, and let C++ report actual
        // size.
        const INITIAL_BUFFER_SIZE: usize = 128;
        let mut public_key_buffer: Vec<u8> = vec![0; INITIAL_BUFFER_SIZE];
        let mut public_key_len: usize = public_key_buffer.len();

        // Call the C++ function via the FFI pointer.
        // This is unsafe because we are calling external C code.
        let success = unsafe {
            (self.verify_jwt_cb)(
                self.verifier_context.0, // Pass the raw C++ object pointer.
                assertion.content.as_ptr(),
                assertion.content.len(),
                public_key_buffer.as_mut_ptr(),
                &mut public_key_len, // Pass mutable reference to length.
            )
        };

        if success {
            eprintln!("---> Rust: C++ verify_jwt_f returned success.");

            // Resize buffer to the actual key length reported by C++.
            public_key_buffer.truncate(public_key_len);

            // Create the struct that holds the verified assertion and the key needed for
            // binding.
            Ok(Box::new(FfiVerifiedAssertion {
                assertion: assertion.clone(), // Clone assertion data for ownership.
                verifier_context: self.verifier_context,
                public_key: public_key_buffer, // Store the extracted public key.
            }))
        } else {
            // Check if failure was due to insufficient buffer size, provide a specific
            // error.
            if public_key_len > INITIAL_BUFFER_SIZE {
                eprintln!(
                    "---> Rust: C++ verify_jwt_f failed: public key buffer too small. Need {} bytes.",
                    public_key_len
                );
                return Err(BoundAssertionVerificationError::GenericFailure {
                    error_msg: format!(
                        "Public key output buffer too small. Need {} bytes, have {}.",
                        public_key_len, INITIAL_BUFFER_SIZE
                    ),
                });
            }

            // Otherwise, it was a general verification or policy failure reported by C++.
            eprintln!("---> Rust: C++ verify_jwt_f returned failure (likely policy violation or invalid JWT).");
            Err(BoundAssertionVerificationError::GenericFailure {
                error_msg: "C++ JWT verification failed (check C++ logs for details)".to_string(),
            })
        }
    }
}

// --- FFI Export ---

/// Creates an Oak ClientSession configuration object using FFI callbacks.
/// This is the main entry point called by the C++ client.
///
/// # Safety
///
/// This function is unsafe because it deals with raw pointers
/// (`verifier_context`) passed from C++. The caller must ensure that
/// `verifier_context` points to a valid object with a lifetime that exceeds the
/// usage of the returned `SessionConfig`.
#[no_mangle]
pub extern "C" fn create_client_session_config(
    verifier_context: *mut c_void,    // Opaque C++ verifier context.
    verify_jwt_cb: VerifyJwtCallback, // C++ JWT verification callback.
) -> *mut SessionConfig {
    eprintln!("---> Rust: create_client_session_config called.");

    // Create the verifier implementation that wraps the C++ callbacks.
    let ffi_verifier: Arc<dyn BoundAssertionVerifier> =
        Arc::new(FfiVerifier { verifier_context: SendSyncRawPtr(verifier_context), verify_jwt_cb });

    // Configure the Oak Session.
    let config =
        SessionConfig::builder(AttestationType::PeerUnidirectional, HandshakeType::NoiseNN)
            // Register our custom verifier implementation. "custom_assertion" must match the key
            // used by the server in its `attest_response`.
            .add_peer_assertion_verifier_ref("custom_assertion".to_string(), &ffi_verifier)
            // Use the simple PassThrough aggregator, as we handle only one assertion.
            .set_assertion_attestation_aggregator(Box::new(PassThrough {}))
            .build();

    eprintln!("---> Rust: ClientSessionConfig created.");
    // Box the config and return a raw pointer, transferring ownership to the C++
    // caller.
    Box::into_raw(Box::new(config))
}
