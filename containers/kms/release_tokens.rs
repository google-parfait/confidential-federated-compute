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

//! Provides functions for verifying and decrypting release tokens, including
//! generating the transform signing key endorsement used to establish the
//! provenance of a release token.

use anyhow::{ensure, Context};
use bssl_crypto::{ec, ecdsa};
use bssl_utils::asn1_signature_to_p1363;
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana::{Algorithm, EllipticCurve},
    CborSerializable, CoseKeyBuilder, CoseSign1Builder, HeaderBuilder,
};
use key_derivation::PUBLIC_KEY_CLAIM;

/// Generates the signing key endorsement for a transform.
///
/// The endorsement is a CBOR Web Token (CWT; RFC 8392) signed by the cluster
/// key. It contains the transform signing key in a claim; the caller may
/// provide additional claims as desired (e.g. information about how the
/// transform matched the access policy).
pub fn endorse_transform_signing_key(
    transform_signing_key: &[u8],
    cluster_key: &ecdsa::PrivateKey<ec::P256>,
    mut claims: ClaimsSet,
) -> anyhow::Result<Vec<u8>> {
    // Add a claim containing the transform signing key. An uncompressed X9.62
    // public key is "0x04<x><y>". For P-256, x and y are 32 bytes each.
    ensure!(
        transform_signing_key.starts_with(b"\x04") && transform_signing_key.len() == 65,
        "transform_signing_key is not a X9.62-encoded ECDSA P-256 public key"
    );
    let cose_key = CoseKeyBuilder::new_ec2_pub_key(
        EllipticCurve::P_256,
        transform_signing_key[1..33].into(),
        transform_signing_key[33..].into(),
    )
    .algorithm(Algorithm::ES256)
    .build()
    .to_vec()
    .map_err(anyhow::Error::msg)
    .context("failed to encode CoseKey")?;
    claims.rest.push((ClaimName::PrivateUse(PUBLIC_KEY_CLAIM), Value::from(cose_key)));

    CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(Algorithm::ES256).build())
        .payload(claims.to_vec().map_err(anyhow::Error::msg).context("failed to encode ClaimsSet")?)
        .try_create_signature(b"", |msg| {
            asn1_signature_to_p1363(&cluster_key.sign(msg))
                .context("failed to convert signature to P1363")
                .map(Into::into)
        })?
        .build()
        .to_vec()
        .map_err(anyhow::Error::msg)
        .context("failed to encode CoseSign1")
}
