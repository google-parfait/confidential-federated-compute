// Copyright 2025 The Trusted Computations Platform Authors.
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

//! Utility functions for working with boringssl, and especially bssl-sys.
//! If a function can be implemented using higher-level (non-unsafe) APIs,
//! prefer to place it in a different module.

/// RAII wrapper for bssl_sys::BIGNUM.
struct Bignum(*mut bssl_sys::BIGNUM);

impl Bignum {
    pub fn new(value: &[u8]) -> Self {
        // SAFETY: `BN_bin2bn` will only access the first `value.len()` bytes of
        // `value.as_ptr()`, which are guaranteed to be allocated and
        // initialized.
        let ptr = unsafe { bssl_sys::BN_bin2bn(value.as_ptr(), value.len(), std::ptr::null_mut()) };
        // `ptr` is only NULL if allocation failed, which we don't handle.
        assert!(!ptr.is_null());
        Self(ptr)
    }
}

impl Drop for Bignum {
    fn drop(&mut self) {
        // SAFETY: `self.0` points to memory allocated by boringssl.
        unsafe { bssl_sys::BN_free(self.0) }
    }
}

/// RAII wrapper for bssl_sys::ECDSA_SIG.
struct EcdsaSignature(*mut bssl_sys::ECDSA_SIG);

impl EcdsaSignature {
    pub fn new() -> Self {
        // SAFETY: No input requirements.
        let ptr = unsafe { bssl_sys::ECDSA_SIG_new() };
        // `ptr` is only NULL if allocation failed, which we don't handle.
        assert!(!ptr.is_null());
        Self(ptr)
    }

    /// Creates a signature from a DER-encoded ECDSA-Sig-Value structure
    /// (ASN.1 DER).
    pub fn from_bytes(value: &[u8]) -> Option<Self> {
        // SAFETY: `ECDSA_SIG_from_bytes` will only access the first
        // `value.len()` bytes of `value.as_ptr()`, which are guaranteed to be
        // allocated and initialized.
        let ptr = unsafe { bssl_sys::ECDSA_SIG_from_bytes(value.as_ptr(), value.len()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }
}

impl Drop for EcdsaSignature {
    fn drop(&mut self) {
        // SAFETY: `self.0` points to memory allocated by boringssl.
        unsafe { bssl_sys::ECDSA_SIG_free(self.0) }
    }
}

/// Converts a ECDSA signature from ASN.1 DER (RFC 5912) to IEEE P1363 (raw r|s)
/// as used by COSE (RFC 9053 section 2.1).
pub fn asn1_signature_to_p1363(signature: &[u8]) -> Option<[u8; 64]> {
    let sig = EcdsaSignature::from_bytes(signature)?;

    let mut bn_r = std::ptr::null();
    let mut bn_s = std::ptr::null();
    // SAFETY: `sig.0` points to an initialized ECDSA_SIG object.
    // `ECDSA_SIG_get0` will set `bn_r` and `bn_s` to non-NULL pointers to
    // BIGNUM objects owned by the ECDSA_SIG.
    unsafe { bssl_sys::ECDSA_SIG_get0(sig.0, &mut bn_r, &mut bn_s) };
    assert!(!bn_r.is_null() && !bn_s.is_null());

    let mut signature = [0; 64];
    let (r, s) = signature.split_at_mut(32);
    // SAFETY: `bn_r` and `bn_s` point to initialized BIGNUM objects, and `sig`
    // is still in scope. On success, `BN_bn2bin_padded` will write to exactly
    // `r.len()`/`s.len()` bytes of `r`/`s`.
    if unsafe { bssl_sys::BN_bn2bin_padded(r.as_mut_ptr(), r.len(), bn_r) } != 1
        || unsafe { bssl_sys::BN_bn2bin_padded(s.as_mut_ptr(), s.len(), bn_s) } != 1
    {
        return None;
    }
    Some(signature)
}

/// Converts an ECDSA signature from IEEE P1363 (raw r|s, as used by COSE) to
/// ASN.1 DER (as used by bssl).
pub fn p1363_signature_to_asn1(signature: &[u8; 64]) -> Vec<u8> {
    let (r, s) = signature.split_at(32);
    let bn_r = Bignum::new(r);
    let bn_s = Bignum::new(s);
    let sig = EcdsaSignature::new();
    // SAFETY: `sig.0`, `bn_r.0`, and `bn_s.0` all point to initialized objects.
    // `ECDSA_SIG_set0` takes ownership of `bn_r.0` and `bn_s.0` on success.
    assert_eq!(unsafe { bssl_sys::ECDSA_SIG_set0(sig.0, bn_r.0, bn_s.0) }, 1);
    std::mem::forget(bn_r);
    std::mem::forget(bn_s);

    let mut out_bytes = std::ptr::null_mut();
    let mut out_len = 0;
    // SAFETY: `sig.0` points to an initialized ECDSA_SIG object. On success,
    // `out_bytes` will point to a buffer containing at least `out_len`
    // initialized bytes; it must be freed using `bssl_sys::OPENSSL_free`.
    assert_eq!(unsafe { bssl_sys::ECDSA_SIG_to_bytes(&mut out_bytes, &mut out_len, sig.0) }, 1);

    // SAFETY: `out_bytes` is valid and contains at least `out_len` bytes. Since
    // the element type is `u8`, we don't need to worry about alignment.
    let signature = unsafe { std::slice::from_raw_parts(out_bytes, out_len) }.to_vec();
    // SAFETY: `out_bytes` points to a buffer allocated by boringssl.
    unsafe { bssl_sys::OPENSSL_free(out_bytes as *mut std::ffi::c_void) };
    signature
}
