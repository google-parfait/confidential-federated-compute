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

//! Utility functions for working with boringssl.

// TODO: b/398874186 - Switch from p256 to boringssl.
use p256::ecdsa::Signature;

/// Converts a ECDSA signature from ASN.1 DER (RFC 5912) to IEEE P1363 (raw r|s)
/// as used by COSE (RFC 9053 section 2.1).
pub fn asn1_signature_to_p1363(signature: &[u8]) -> Option<Vec<u8>> {
    Some(Signature::from_der(signature).ok()?.to_bytes().to_vec())
}

/// Converts a ECDSA signature from IEEE P1363 (raw r|s, as used by COSE) to
/// ASN.1 DER (as used by bssl).
pub fn p1363_signature_to_asn1(signature: &[u8]) -> Option<Vec<u8>> {
    Some(Signature::from_slice(signature).ok()?.to_der().to_bytes().to_vec())
}
