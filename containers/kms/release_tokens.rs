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

use anyhow::{anyhow, ensure, Context};
use bssl_crypto::{ec, ecdsa};
use bssl_utils::{asn1_signature_to_p1363, p1363_signature_to_asn1};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana, Algorithm, CborSerializable, CoseEncrypt0, CoseKey, CoseKeyBuilder, CoseSign1,
    CoseSign1Builder, HeaderBuilder, KeyType, Label,
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
        iana::EllipticCurve::P_256,
        transform_signing_key[1..33].into(),
        transform_signing_key[33..].into(),
    )
    .algorithm(iana::Algorithm::ES256)
    .build()
    .to_vec()
    .map_err(anyhow::Error::msg)
    .context("failed to encode CoseKey")?;
    claims.rest.push((ClaimName::PrivateUse(PUBLIC_KEY_CLAIM), Value::from(cose_key)));

    CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(claims.to_vec().map_err(anyhow::Error::msg).context("failed to encode ClaimsSet")?)
        .try_create_signature(b"", |msg| {
            asn1_signature_to_p1363(&cluster_key.sign(msg))
                .context("failed to convert signature to P1363")
        })?
        .build()
        .to_vec()
        .map_err(anyhow::Error::msg)
        .context("failed to encode CoseSign1")
}

/// Verifies that a release token is valid and was signed by the signing key in
/// the endorsement.
///
/// Returns the release token's payload, the endorsement's claims, and a
/// function that verifies the endorsement's signature.
pub fn verify_release_token(
    token: &[u8],
    endorsement: &[u8],
) -> anyhow::Result<(
    CoseEncrypt0,
    ClaimsSet,
    impl Fn(&ecdsa::PublicKey<ec::P256>) -> anyhow::Result<()>,
)> {
    // Extract the transform's public key from the endorsement.
    let endorsement = CoseSign1::from_slice(endorsement)
        .map_err(anyhow::Error::msg)
        .context("failed to parse endorsement")?;
    let claims = ClaimsSet::from_slice(endorsement.payload.as_deref().unwrap_or_default())
        .map_err(anyhow::Error::msg)
        .context("failed to parse endorsement claims")?;
    let cose_key = claims
        .rest
        .iter()
        .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
        .and_then(|(_, value)| value.as_bytes())
        .context("endorsement is missing public key")?;
    let cose_key = CoseKey::from_slice(cose_key)
        .map_err(anyhow::Error::msg)
        .context("failed to parse public key")?;
    ensure!(cose_key.kty == KeyType::Assigned(iana::KeyType::EC2), "unsupported public key type");
    ensure!(
        cose_key.alg == Some(Algorithm::Assigned(iana::Algorithm::ES256)),
        "unsupported public key algorithm"
    );
    let (mut crv, mut x, mut y) = (None, None, None);
    for (label, value) in cose_key.params {
        use iana::Ec2KeyParameter;
        match (label, value) {
            (Label::Int(l), v) if l == Ec2KeyParameter::Crv as i64 => crv = Some(v),
            (Label::Int(l), Value::Bytes(v)) if l == Ec2KeyParameter::X as i64 => x = Some(v),
            (Label::Int(l), Value::Bytes(v)) if l == Ec2KeyParameter::Y as i64 => y = Some(v),
            _ => {}
        }
    }
    ensure!(crv == Some(Value::from(iana::EllipticCurve::P_256 as i64)), "unsupported curve");
    ensure!(x.as_ref().is_some_and(|x| x.len() == 32), "invalid x coordinate");
    ensure!(y.as_ref().is_some_and(|y| y.len() == 32), "invalid y coordinate");
    let x962_public_key = [b"\x04", x.unwrap().as_slice(), y.unwrap().as_slice()].concat();
    let public_key = ecdsa::PublicKey::<ec::P256>::from_x962_uncompressed(&x962_public_key)
        .context("failed to parse public key")?;

    // Verify the release token's signature.
    let token = CoseSign1::from_slice(token)
        .map_err(anyhow::Error::msg)
        .context("failed to parse release token")?;
    ensure!(token.protected.header.alg == cose_key.alg, "release token algorithm mismatch");
    token
        .verify_signature(b"", |signature, data| {
            let signature =
                p1363_signature_to_asn1(signature).context("ASN.1 conversion failed")?;
            public_key
                .verify(data, &signature)
                .map_err(|_| anyhow!("signature verification failed"))
        })
        .context("invalid release token signature")?;

    // Define a function to verify the endorsement's signature.
    ensure!(
        endorsement.protected.header.alg == Some(Algorithm::Assigned(iana::Algorithm::ES256)),
        "unsupported endorsement signature algorithm"
    );
    let verify_signature_fn = move |cluster_key: &ecdsa::PublicKey<ec::P256>| {
        endorsement
            .verify_signature(b"", |signature, data| {
                let signature =
                    p1363_signature_to_asn1(signature).context("ASN.1 conversion failed")?;
                cluster_key
                    .verify(data, &signature)
                    .map_err(|_| anyhow!("signature verification failed"))
            })
            .context("invalid endorsement signature")
    };

    let token_payload = CoseEncrypt0::from_slice(token.payload.as_deref().unwrap_or_default())
        .map_err(anyhow::Error::msg)
        .context("invalid release token payload")?;
    Ok((token_payload, claims, verify_signature_fn))
}
