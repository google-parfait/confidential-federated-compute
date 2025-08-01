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

use std::cmp::min;

use anyhow::{anyhow, bail, ensure, Context};
use bssl_crypto::{hkdf, hpke};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana, Algorithm, CborSerializable, CoseKey, CoseSign1Builder, Header, KeyOperation, KeyType,
    Label,
};
use futures::stream::{FuturesOrdered, TryStreamExt};
use oak_sdk_containers::Signer;

// Private CWT claims; see
// https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/cbor_ids.md.
pub const PUBLIC_KEY_CLAIM: i64 = -65537;
pub const ACCESS_POLICY_SHA256_CLAIM: i64 = -65543;

// Private CoseKey algorithms; see
// https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/cbor_ids.md.
pub const HPKE_BASE_X25519_SHA256_AES128GCM: i64 = -65537;

/// Derives a private key, encoded as a CoseKey, for each HKDF info value.
pub fn derive_private_keys(
    alg: i64,
    key_id: &[u8],
    ikm: &[u8],
    infos: impl IntoIterator<Item = impl AsRef<[u8]>> + Copy,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let private_keys = derive_raw_private_keys(alg, ikm, infos)?;
    infos
        .into_iter()
        .zip(private_keys)
        .map(|(info, private_key)| {
            build_cose_key(alg, key_id, info.as_ref(), /* public= */ false, private_key)
        })
        .collect()
}

/// Derives a public key, encoded as a CoseKey, for each HKDF info value.
pub fn derive_public_keys(
    alg: i64,
    key_id: &[u8],
    ikm: &[u8],
    infos: impl IntoIterator<Item = impl AsRef<[u8]>> + Copy,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let private_keys = derive_raw_private_keys(alg, ikm, infos)?;
    infos
        .into_iter()
        .zip(private_keys)
        .map(|(info, private_key)| {
            let public_key = get_public_key(alg, &private_key)?;
            build_cose_key(alg, key_id, info.as_ref(), /* public= */ true, public_key)
        })
        .collect()
}

/// Derives a public key, encoded as a CWT, for each access policy hash.
pub async fn derive_public_cwts<S: Signer>(
    alg: i64,
    key_id: &[u8],
    ikm: &[u8],
    claims: ClaimsSet,
    access_policy_hashes: impl IntoIterator<Item = impl AsRef<[u8]>> + Copy,
    signer: &S,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut cwts = FuturesOrdered::new();
    let cose_keys = derive_public_keys(alg, key_id, ikm, access_policy_hashes)?;
    for (access_policy_hash, cose_key) in access_policy_hashes.into_iter().zip(cose_keys) {
        let mut claims = claims.clone();
        claims.rest.push((ClaimName::PrivateUse(PUBLIC_KEY_CLAIM), Value::from(cose_key)));
        claims.rest.push((
            ClaimName::PrivateUse(ACCESS_POLICY_SHA256_CLAIM),
            Value::Bytes(access_policy_hash.as_ref().to_vec()),
        ));
        cwts.push_back(build_cwt(claims, signer));
    }
    cwts.try_collect().await
}

/// Derives a raw private key for each HKDF info value.
fn derive_raw_private_keys(
    alg: i64,
    ikm: &[u8],
    infos: impl IntoIterator<Item = impl AsRef<[u8]>>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    ensure!(alg == HPKE_BASE_X25519_SHA256_AES128GCM, "unsupported algorithm: {}", alg);
    // Per RFC 5869, the salt should be random and at least as long as the hash
    // (32 bytes). If new algorithms are added, they should use their own salts.
    let salt = hkdf::Salt::NonEmpty(&[
        5, 36, 164, 198, 227, 192, 203, 187, 132, 228, 115, 85, 125, 175, 222, 66, 105, 238, 84,
        27, 39, 75, 83, 205, 219, 139, 185, 111, 109, 255, 189, 228,
    ]);
    let prk = hkdf::HkdfSha256::extract(ikm, salt);
    Ok(infos
        .into_iter()
        .map(|info| prk.expand::<32>(info.as_ref()))
        .map(|mut secret| {
            // To catch incorrect curve implementations, explicitly unset/set
            // the bits that should be set/unset by X25519 scalar multiplication
            // (RFC 7748 section 5). This ensures keys match those generated by
            // boringssl:
            // https://github.com/google/boringssl/blob/447dc9dcaedb961426ce7f51209fd019b93e6e77/crypto/curve25519/curve25519.cc#L2109-L2111.
            secret[0] |= !248;
            secret[31] &= !64;
            secret[31] |= !127;
            secret.into()
        })
        .collect())
}

/// Gets the public key for a private key.
fn get_public_key(alg: i64, private_key: &[u8]) -> anyhow::Result<Vec<u8>> {
    match alg {
        HPKE_BASE_X25519_SHA256_AES128GCM => hpke::Kem::X25519HkdfSha256
            .public_from_private(private_key)
            .ok_or_else(|| anyhow!("derived private key is invalid")),
        _ => bail!("unsupported algorithm: {}", alg),
    }
}

/// Encodes a public or private key as a CoseKey.
fn build_cose_key(
    alg: i64,
    key_id: &[u8],
    info: &[u8],
    public: bool,
    key: Vec<u8>,
) -> anyhow::Result<Vec<u8>> {
    ensure!(alg == HPKE_BASE_X25519_SHA256_AES128GCM, "unsupported algorithm: {}", alg);
    let key_param = if public { iana::OkpKeyParameter::X } else { iana::OkpKeyParameter::D };
    let key_op = if public { iana::KeyOperation::Encrypt } else { iana::KeyOperation::Decrypt };
    CoseKey {
        kty: KeyType::Assigned(iana::KeyType::OKP),
        // Use the key_id + the first 4 bytes of the info field key as a
        // likely-unique key id. Including the key_id allows the resulting key
        // to be mapped back to the original keyset key.
        key_id: [key_id, &info[0..min(4, info.len())]].concat(),
        alg: Some(Algorithm::PrivateUse(alg)),
        key_ops: [KeyOperation::Assigned(key_op)].into(),
        params: vec![
            (
                Label::Int(iana::OkpKeyParameter::Crv as i64),
                Value::from(iana::EllipticCurve::X25519 as u64),
            ),
            (Label::Int(key_param as i64), Value::Bytes(key)),
        ],
        ..Default::default()
    }
    .to_vec()
    .map_err(anyhow::Error::msg)
    .context("failed to encode CoseKey")
}

/// Encodes a public key as a CWT.
async fn build_cwt<S: Signer>(claims: ClaimsSet, signer: &S) -> anyhow::Result<Vec<u8>> {
    // CoseSign1Builder.create_signature doesn't support async signing, so
    // we instead collect the bytes to be signed, then generate and attach
    // the signature.
    let mut to_be_signed = Vec::with_capacity(0);
    CoseSign1Builder::new()
        .protected(Header {
            alg: Some(Algorithm::Assigned(iana::Algorithm::ES256)),
            ..Default::default()
        })
        .payload(claims.to_vec().map_err(anyhow::Error::msg).context("failed to encode ClaimsSet")?)
        .create_signature(b"", |msg| {
            to_be_signed.extend_from_slice(msg);
            Vec::with_capacity(0)
        })
        // The Signer produces raw `r|s` signatures, so no conversion is needed.
        .signature(
            signer.sign(to_be_signed.as_slice()).await.context("failed to sign CWT")?.signature,
        )
        .build()
        .to_vec()
        .map_err(anyhow::Error::msg)
        .context("failed to encode CWT")
}
