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

//! Rust FFI module for creating an Oak ServerSession configuration and managing
//! keys.
//!
//! Provides C FFI functions:
//! - `generate_key_pair`: Creates an ECDSA P-256 key pair.
//! - `create_server_session_config`: Builds the Oak SessionConfig for the
//!   server, including a custom assertion generator that binds the session ID
//!   using the generated private key.

use anyhow::Result;
use oak_proto_rust::oak::{attestation::v1::Assertion, session::v1::SessionBinding};
use oak_session::{
    attestation::AttestationType,
    config::SessionConfig,
    generator::{BindableAssertion, BindableAssertionGenerator, BindableAssertionGeneratorError},
    handshake::HandshakeType,
};
// Import crypto crates for P-256 ECDSA signing.
use p256::ecdsa::{
    signature::{rand_core::OsRng, Signer}, // Use the `Signer` trait for signing.
    Signature as EcdsaSignature,           // The signature type.
    SigningKey,                            // The private key type.
};
use std::{ffi::c_char, ptr, slice, str::Utf8Error, sync::Arc};

// --- FFI Type Definitions ---

/// Opaque handle type for the ECDSA P-256 private key.
/// C++ only sees this as `oak::session::SigningKeyHandle*`.
/// The underlying Rust type is `p256::ecdsa::SigningKey`.
pub type SigningKeyHandle = SigningKey;

// --- Key Management (FFI Export) ---

/// Generates a P-256 key pair via FFI.
///
/// Returns an opaque handle to the private key (`SigningKey`) and writes the
/// raw public key bytes (SEC1 uncompressed format) into the provided C++
/// buffer.
///
/// # Safety
///
/// This function is unsafe because it dereferences raw pointers passed from
/// C++. The caller *must* ensure:
/// - `out_public_key_bytes` points to a valid buffer of at least
///   `public_key_capacity` bytes.
/// - `public_key_capacity` is large enough (at least 65 bytes for P-256
///   uncompressed).
/// - `out_private_key_handle` points to a valid `*mut SigningKeyHandle`.
/// Ownership of the returned `SigningKeyHandle` is transferred to the C++
/// caller.
#[no_mangle]
pub extern "C" fn generate_key_pair(
    out_public_key_bytes: *mut u8,
    public_key_capacity: usize,
    out_private_key_handle: *mut *mut SigningKeyHandle,
) -> i32 {
    eprintln!("---> Rust: generate_key_pair called.");
    // Generate a new random signing key.
    let signing_key = SigningKey::random(&mut OsRng);
    // Derive the corresponding public key.
    let verifying_key = signing_key.verifying_key();

    // Serialize the public key to SEC1 uncompressed format (65 bytes for P-256).
    let public_key_sec1 = verifying_key.to_encoded_point(/* compress= */ false);
    let public_key_bytes = public_key_sec1.as_bytes();
    let key_len = public_key_bytes.len();

    // Check if the C++ buffer is large enough.
    if public_key_capacity < key_len {
        eprintln!(
            "---> Rust ERROR: Public key buffer too small (need {}, have {})",
            key_len, public_key_capacity
        );
        return -1; // Indicate error.
    }

    // Unsafe block to write to C++ pointers.
    unsafe {
        // Copy the public key bytes into the C++ buffer.
        ptr::copy_nonoverlapping(public_key_bytes.as_ptr(), out_public_key_bytes, key_len);
        // Box the signing key (heap allocation) and return the raw pointer as the
        // handle. `Box::into_raw` transfers ownership to C++.
        *out_private_key_handle = Box::into_raw(Box::new(signing_key));
    }

    eprintln!("---> Rust: Generated key pair, public key size: {}", key_len);
    key_len as i32 // Return the number of bytes written.
}

// --- Custom Attestation Logic ---

/// Custom implementation of `BindableAssertion`.
/// Holds the assertion data (JWT) and the signing key needed for binding.
struct CustomAssertion {
    /// The assertion proto containing the JWT.
    assertion: Assertion,
    /// Arc-wrapped signing key for thread-safe sharing and binding.
    signing_key: Arc<SigningKey>,
}

impl BindableAssertion for CustomAssertion {
    /// Returns a reference to the assertion proto.
    fn assertion(&self) -> &Assertion {
        eprintln!("---> Rust: CustomAssertion.assertion() called.");
        &self.assertion
    }

    /// Creates the session binding signature.
    /// Signs the provided `session_id` using the stored private key.
    /// The `Signer` trait implementation handles hashing internally.
    fn bind(
        &self,
        session_id: &[u8], // The Noise session ID provided by Oak Session.
    ) -> Result<SessionBinding, BindableAssertionGeneratorError> {
        eprintln!("---> Rust: CustomAssertion.bind() called (signing session_id).");
        let signing_key = &self.signing_key;

        // Sign the session_id directly using the Signer trait.
        // This implicitly performs the required hashing (e.g., SHA-256 for P-256).
        let signature: EcdsaSignature = signing_key.sign(session_id);

        // Package the raw signature bytes into the SessionBinding proto.
        Ok(SessionBinding { binding: signature.to_vec(), ..Default::default() })
    }
}

/// Custom implementation of `BindableAssertionGenerator`.
/// Responsible for creating `CustomAssertion` instances when requested by Oak
/// Session.
struct CustomAssertionGenerator {
    /// The JWT attestation token received from C++.
    token: String,
    /// Arc-wrapped signing key (cloned into `CustomAssertion`).
    signing_key: Arc<SigningKey>,
}

impl BindableAssertionGenerator for CustomAssertionGenerator {
    /// Generates a `CustomAssertion` instance containing the token and signing
    /// key.
    fn generate(
        &self,
    ) -> Result<Box<dyn BindableAssertion + 'static>, BindableAssertionGeneratorError> {
        eprintln!("---> Rust: CustomAssertionGenerator.generate() called.");
        // Create the Assertion proto with the JWT content.
        let assertion = Assertion { content: self.token.as_bytes().to_vec() };
        // Create the CustomAssertion, cloning the Arc for shared ownership of the key.
        Ok(Box::new(CustomAssertion { assertion, signing_key: self.signing_key.clone() }))
    }
}

// --- FFI Helper ---

/// Safely converts a C-style string (`*const c_char`, `len`) to a Rust
/// `String`.
fn c_str_to_string(ptr: *const c_char, len: usize) -> Result<String, Utf8Error> {
    // Unsafe block to create a slice from the raw C pointer.
    let bytes = unsafe { slice::from_raw_parts(ptr as *const u8, len) };
    // Attempt to convert the byte slice to a UTF-8 string.
    std::str::from_utf8(bytes).map(String::from)
}

// --- Server Config Creation (FFI Export) ---

/// Creates a server session configuration object via FFI.
/// Takes the attestation token and the private key handle from C++, sets up
/// the custom assertion generator, and returns an opaque pointer to the config.
///
/// # Safety
///
/// This function is unsafe because it deals with raw pointers passed from C++.
/// - It dereferences `attestation_token` assuming it's valid for
///   `attestation_token_len` bytes.
/// - It takes ownership of `private_key_handle` using `Box::from_raw`, assuming
///   it's a valid pointer previously returned by `generate_key_pair` and that
///   C++ has relinquished ownership. Double-free or use-after-free will occur
///   if C++ mismanages the handle.
#[no_mangle]
pub extern "C" fn create_server_session_config(
    attestation_token: *const c_char,
    attestation_token_len: usize,
    private_key_handle: *mut SigningKeyHandle, // Opaque private key handle from C++.
) -> *mut SessionConfig {
    // Convert the C string token to a Rust String. Panics if invalid UTF-8.
    let token_str = c_str_to_string(attestation_token, attestation_token_len)
        .expect("Invalid UTF-8 token from C++");

    eprintln!("---> Rust: Received token (len {}).", token_str.len());
    eprintln!("---> Rust: Received private key handle: {:?}", private_key_handle);

    // Take ownership of the private key handle passed from C++.
    // Wrap it in an Arc immediately for safe sharing with the assertion generator.
    let signing_key_arc = unsafe {
        if private_key_handle.is_null() {
            // Check for null pointers from C++.
            panic!("Received null private key handle from C++");
        }
        // Reconstruct the Box from the raw pointer to manage its lifetime in Rust.
        let key_box = Box::from_raw(private_key_handle);
        // Convert the Box into an Arc for shared ownership.
        Arc::from(key_box)
    };

    // Create the custom assertion generator instance.
    let generator: Arc<dyn BindableAssertionGenerator> =
        Arc::new(CustomAssertionGenerator { token: token_str, signing_key: signing_key_arc });

    // Build the Oak ServerSession configuration.
    let config =
        SessionConfig::builder(AttestationType::SelfUnidirectional, HandshakeType::NoiseNN)
            // Register our custom generator. "custom_assertion" must match the key
            // expected by the client verifier.
            .add_self_assertion_generator_ref("custom_assertion".to_string(), &generator)
            .build();

    eprintln!("---> Rust: ServerSessionConfig created.");
    // Box the config and return a raw pointer, transferring ownership to C++.
    Box::into_raw(Box::new(config))
}
