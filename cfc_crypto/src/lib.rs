// Copyright 2023 Google LLC.
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

#![no_std]

extern crate alloc;

use aes_gcm_siv::{
    aead::{Aead, OsRng, Payload},
    Aes128GcmSiv, KeyInit,
};
use alloc::{vec, vec::Vec};
use anyhow::anyhow;
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana, Algorithm, CborSerializable, CoseKey, CoseSign1, KeyType, Label,
};
use hpke::{
    aead::AesGcm128, kdf::HkdfSha256, kem::X25519HkdfSha256, Deserializable, Kem, OpModeR, OpModeS,
    Serializable,
};

pub type PrivateKey = <X25519HkdfSha256 as Kem>::PrivateKey;

// A fixed random nonce, which is safe to reuse because the symmetric key is
// never reused.
static NONCE: [u8; 12] = [0x74, 0xDF, 0x8F, 0xD4, 0xBE, 0x34, 0xAF, 0x64, 0x7F, 0x5E, 0x54, 0xF6];

// The HPKE info field is not used.
static INFO: [u8; 0] = [];

// Private CWT claims; see
// https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/cbor_ids.md.
pub const PUBLIC_KEY_CLAIM: i64 = -65537;
pub const CONFIG_PROPERTIES_CLAIM: i64 = -65538;

// Private CoseKey algorithms; see
// https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/cbor_ids.md.
const HPKE_BASE_X25519_SHA256_AES128GCM: i64 = -65537;
const AEAD_AES_128_GCM_SIV_FIXED_NONCE: i64 = -65538;

/// Wraps a symmetric encryption key using HPKE.
///
/// # Return Value
///
/// Returns `Ok((encapped_key, encrypted_symmetric_key))` on success.
fn wrap_symmetric_key(
    symmetric_key: &[u8],
    recipient_public_key: &CoseKey,
    associated_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    // Check that the CoseKey can be used for rewrapping.
    if recipient_public_key.kty != KeyType::Assigned(iana::KeyType::OKP)
        || recipient_public_key.alg
            != Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM))
        || !recipient_public_key.params.iter().any(|(label, value)| {
            label == &Label::Int(iana::OkpKeyParameter::Crv as i64)
                && value == &Value::from(iana::EllipticCurve::X25519 as u64)
        })
    {
        return Err(anyhow!("unsupported CoseKey type"));
    }

    // Extract the raw public key and convert it to a PublicKey.
    let raw_recipient_public_key = recipient_public_key
        .params
        .iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
        .and_then(|(_, value)| value.as_bytes())
        .ok_or_else(|| anyhow!("CoseKey missing X parameter"))?;
    let public_key = <X25519HkdfSha256 as Kem>::PublicKey::from_bytes(raw_recipient_public_key)
        .map_err(|err| anyhow!("failed to parse recipient public key: {:?}", err))?;

    // Rewrap the symmetric key.
    let (encapped_key, encrypted_symmetric_key) =
        hpke::single_shot_seal::<AesGcm128, HkdfSha256, X25519HkdfSha256, _>(
            &OpModeS::Base,
            &public_key,
            &INFO,
            symmetric_key,
            associated_data,
            &mut OsRng,
        )
        .map_err(|err| anyhow!("failed to seal key: {:?}", err))?;
    Ok((encapped_key.to_bytes().to_vec(), encrypted_symmetric_key))
}

/// Unwraps a symmetric encryption that was wrapped using `wrap_symmetric_key`.
///
/// # Return Value
///
/// Returns `Ok(symmetric_key)` on success.
fn unwrap_symmetric_key(
    encrypted_symmetric_key: &[u8],
    serialized_encapped_key: &[u8],
    private_key: &PrivateKey,
    associated_data: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let encapped_key = <X25519HkdfSha256 as Kem>::EncappedKey::from_bytes(serialized_encapped_key)
        .map_err(|err| anyhow!("failed to load encapped key: {:?}", err))?;
    hpke::single_shot_open::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
        &OpModeR::Base,
        private_key,
        &encapped_key,
        &INFO,
        encrypted_symmetric_key,
        associated_data,
    )
    .map_err(|err| anyhow!("failed to unwrap symmetric key: {:?}", err))
}

/// Generates a random keypair.
pub fn gen_keypair(key_id: &[u8]) -> (PrivateKey, CoseKey) {
    let (private_key, raw_public_key) = <X25519HkdfSha256 as Kem>::gen_keypair(&mut OsRng);
    let public_key = CoseKey {
        kty: KeyType::Assigned(iana::KeyType::OKP),
        key_id: key_id.to_vec(),
        alg: Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)),
        params: vec![
            (
                Label::Int(iana::OkpKeyParameter::Crv as i64),
                Value::from(iana::EllipticCurve::X25519 as u64),
            ),
            (
                Label::Int(iana::OkpKeyParameter::X as i64),
                Value::Bytes(raw_public_key.to_bytes().to_vec()),
            ),
        ],
        ..Default::default()
    };
    (private_key, public_key)
}

/// Encrypts client data using a combination of HPKE and AEAD.
///
/// # Arguments
///
/// * `plaintext` - The message to be encrypted.
/// * `public_key` - The public key of the recipient.
/// * `associated_data` - Additional data to be verified along with the message.
///
/// # Return Value
///
/// Returns `Ok((ciphertext, encapped_key, encrypted_symmetric_key))` on
/// success.
pub fn encrypt_message(
    plaintext: &[u8],
    public_key: &CoseKey,
    associated_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    // Encrypt the plaintext using AEAD.
    let symmetric_key = Aes128GcmSiv::generate_key(OsRng);
    let cipher = Aes128GcmSiv::new(&symmetric_key);
    let ciphertext = cipher
        .encrypt((&NONCE).into(), Payload { msg: plaintext, aad: associated_data })
        .map_err(|err| anyhow!("failed to encrypt plaintext: {:?}", err))?;

    // Construct and serialize a CoseKey containing the symmetric key material.
    let cose_key = CoseKey {
        kty: KeyType::Assigned(iana::KeyType::Symmetric),
        alg: Some(Algorithm::PrivateUse(AEAD_AES_128_GCM_SIV_FIXED_NONCE)),
        params: vec![(
            Label::Int(iana::SymmetricKeyParameter::K as i64),
            Value::Bytes(symmetric_key.to_vec()),
        )],
        ..Default::default()
    }
    .to_vec()
    .map_err(|err| anyhow!("failed to serialize CoseKey: {}", err))?;

    // Encrypt the symmetric key using HPKE.
    let (encapped_key, encrypted_symmetric_key) =
        wrap_symmetric_key(&cose_key, public_key, associated_data)?;
    Ok((ciphertext, encapped_key, encrypted_symmetric_key))
}

/// Unwraps and re-wraps the AEAD symmetric key for consumption by another
/// party.
///
/// Instead of directly decrypting the message, this method allows the AEAD
/// symmetric to be passed off to another party without sharing the private key.
/// In other words, it opens the original HPKE message and then re-seals it for
/// the new recipient.
///
/// # Arguments
///
/// * `encrypted_symmetric_key` - The encrypted symmetric key produced by
///   `encrypt_message`.
/// * `serialized_encapped_key` - The encapped public key returned by
///   `encrypt_message`.
/// * `private_key` - The corresponding private key for the public key passed to
///   `encrypt_message`.
/// * `unwrap_associated_data` - The associated data passed to
///   `encrypt_message`.
/// * `recipient_public_key` - The public key of the recipient.
/// * `wrap_associated_data` - Additional data to be verified along with the
///   message. This replaces `unwrap_associated_data`.
///
/// # Return Value
///
/// Returns `Ok((encapped_key, encrypted_symmetric_key))` on success.
pub fn rewrap_symmetric_key(
    encrypted_symmetric_key: &[u8],
    serialized_encapped_key: &[u8],
    private_key: &PrivateKey,
    unwrap_associated_data: &[u8],
    recipient_public_key: &CoseKey,
    wrap_associated_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    // Unwrap the symmetric key using HPKE.
    let symmetric_key = unwrap_symmetric_key(
        encrypted_symmetric_key,
        serialized_encapped_key,
        private_key,
        unwrap_associated_data,
    )?;

    // Re-wrap the symmetric key using HPKE.
    wrap_symmetric_key(&symmetric_key, recipient_public_key, wrap_associated_data)
}

/// Decrypts data produced using `encrypt_message`.
///
/// # Arguments
///
/// * `ciphertext` - The encrypted message.
/// * `ciphertext_associated_data` - The associated data passed to
///   `encrypt_message`.
/// * `encrypted_symmetric_key` - The encrypted symmetric key produced by
///   `rewrap_symmetric_key`.
/// * `encrypted_symmetric_key_associated_data` - The associated data passed to
///   `rewrap_symmetric_key`.
/// * `serialized_encapped_key` - The encapped public key returned by
///   `rewrap_symmetric_key`.
/// * `private_key` - The corresponding private key for the public key passed to
///   `rewrap_symmetric_key`.
///
/// # Return Value
///
/// Return `Ok(plaintext)` on success.
pub fn decrypt_message(
    ciphertext: &[u8],
    ciphertext_associated_data: &[u8],
    encrypted_symmetric_key: &[u8],
    encrypted_symmetric_key_associated_data: &[u8],
    serialized_encapped_key: &[u8],
    private_key: &PrivateKey,
) -> anyhow::Result<Vec<u8>> {
    let symmetric_key = unwrap_symmetric_key(
        encrypted_symmetric_key,
        serialized_encapped_key,
        private_key,
        encrypted_symmetric_key_associated_data,
    )?;

    // Decode the symmetric key and confirm it's of the expected type.
    let cose_key = CoseKey::from_slice(&symmetric_key)
        .map_err(|err| anyhow!("failed to decode CoseKey: {}", err))?;
    if cose_key.kty != KeyType::Assigned(iana::KeyType::Symmetric)
        || cose_key.alg != Some(Algorithm::PrivateUse(AEAD_AES_128_GCM_SIV_FIXED_NONCE))
    {
        return Err(anyhow!("unsupported CoseKey type"));
    }
    let raw_symmetric_key = cose_key
        .params
        .iter()
        .find(|(label, _)| label == &Label::Int(iana::SymmetricKeyParameter::K as i64))
        .and_then(|(_, value)| value.as_bytes())
        .ok_or_else(|| anyhow!("CoseKey missing K parameter"))?;

    let cipher = Aes128GcmSiv::new_from_slice(raw_symmetric_key)
        .map_err(|err| anyhow!("failed to load symmetric key: {:?}", err))?;
    cipher
        .decrypt((&NONCE).into(), Payload { msg: ciphertext, aad: ciphertext_associated_data })
        .map_err(|err| anyhow!("failed to decrypt data: {:?}", err))
}

/// Extracts a CoseKey from a CBOR Web Token (CWT). No validation is performed
/// on the CWT signature or claims.
pub fn extract_key_from_cwt(cwt: &[u8]) -> anyhow::Result<CoseKey> {
    CoseSign1::from_slice(cwt)
        .and_then(|cwt| ClaimsSet::from_slice(cwt.payload.as_deref().unwrap_or_default()))
        .map_err(|err| anyhow!("failed to decode CWT claims: {:?}", err))
        .and_then(|claims| {
            claims
                .rest
                .into_iter()
                .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
                .ok_or_else(|| anyhow!("missing public key claim"))
        })
        .and_then(|(_, value)| {
            CoseKey::from_slice(
                &value
                    .into_bytes()
                    .map_err(|err| anyhow!("invalid public key claim: {:?}", err))?,
            )
            .map_err(|err| anyhow!("failed to decode CoseKey: {:?}", err))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use coset::{cwt::ClaimsSetBuilder, CoseSign1Builder};
    use googletest::prelude::*;

    #[test]
    fn test_gen_keypair_public_key_params() {
        let (_, public_key) = gen_keypair(b"key-id");
        assert_eq!(public_key.kty, KeyType::Assigned(iana::KeyType::OKP));
        assert_eq!(public_key.alg, Some(Algorithm::PrivateUse(HPKE_BASE_X25519_SHA256_AES128GCM)));
        assert_eq!(public_key.key_id, b"key-id");
        assert_eq!(
            public_key
                .params
                .iter()
                .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::Crv as i64))
                .map(|(_, value)| value),
            Some(&Value::from(iana::EllipticCurve::X25519 as u64))
        );
        assert!(public_key
            .params
            .iter()
            .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
            .map(|(_, value)| value)
            .is_some());
    }

    #[test]
    fn test_gen_keypair_is_unique() {
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        assert_ne!(private_key1.to_bytes(), private_key2.to_bytes());
        assert_ne!(public_key1, public_key2);
    }

    #[test]
    fn test_encrypt_rewrap_decrypt() -> anyhow::Result<()> {
        // Encrypt the original message.
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1)?;

        // Rewrap the symmetric key with a different key pair.
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key_id");
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )?;

        // Decrypt it using the second key pair.
        let result = decrypt_message(
            &ciphertext,
            associated_data1,
            &encrypted_symmetric_key2,
            associated_data2,
            &encapped_key2,
            &private_key2,
        )?;
        assert_eq!(result, plaintext);
        Ok(())
    }

    #[test]
    fn test_encrypt_message_with_invalid_public_key() {
        let plaintext = b"plaintext";
        let associated_data = b"associated data";
        let (_, mut public_key) = gen_keypair(b"key-id");
        public_key
            .params
            .iter_mut()
            .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
            .map(|(_, value)| *value = b"invalid".as_slice().into())
            .unwrap();
        assert_that!(
            encrypt_message(plaintext, &public_key, associated_data),
            err(displays_as(contains_substring("failed to parse recipient public key")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_encrypted_symmetric_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, _) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair(b"key-id");
        assert_that!(
            rewrap_symmetric_key(
                b"invalid",
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_encapped_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, _, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair(b"key-id");
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                b"invalid",
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("failed to load encapped key")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_private_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (_, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key2, // Should be private_key1.
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair(b"key-id");
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                b"invalid",
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_public_key_kty() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2.kty = KeyType::Assigned(iana::KeyType::Symmetric);
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("unsupported CoseKey type")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_public_key_alg() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2.alg = Some(Algorithm::Assigned(iana::Algorithm::SHA_256));
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("unsupported CoseKey type")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_missing_public_key_crv() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2
            .params
            .retain(|(label, _)| label != &Label::Int(iana::OkpKeyParameter::Crv as i64));
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("unsupported CoseKey type")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_public_key_crv() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2
            .params
            .iter_mut()
            .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::Crv as i64))
            .map(|(_, value)| *value = Value::from(iana::EllipticCurve::P_256 as i64))
            .unwrap();
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("unsupported CoseKey type")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_missing_public_key_x() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2
            .params
            .retain(|(label, _)| label != &Label::Int(iana::OkpKeyParameter::X as i64));
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("CoseKey missing X parameter")))
        );
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_public_key_x() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, mut public_key2) = gen_keypair(b"key-id");
        public_key2
            .params
            .iter_mut()
            .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
            .map(|(_, value)| *value = b"invalid".as_slice().into())
            .unwrap();
        assert_that!(
            rewrap_symmetric_key(
                &encrypted_symmetric_key,
                &encapped_key,
                &private_key1,
                associated_data1,
                &public_key2,
                associated_data2,
            ),
            err(displays_as(contains_substring("failed to parse recipient public key")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_ciphertext() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (_, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                b"invalid",
                associated_data1,
                &encrypted_symmetric_key2,
                associated_data2,
                &encapped_key2,
                &private_key2,
            ),
            err(displays_as(contains_substring("failed to decrypt data")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_ciphertext_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                &ciphertext,
                b"invalid",
                &encrypted_symmetric_key2,
                associated_data2,
                &encapped_key2,
                &private_key2,
            ),
            err(displays_as(contains_substring("failed to decrypt data")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_encrypted_symmetric_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        let (encapped_key2, _) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                &ciphertext,
                associated_data1,
                b"invalid",
                associated_data2,
                &encapped_key2,
                &private_key2,
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_encrypted_symmetric_key_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                &ciphertext,
                associated_data1,
                &encrypted_symmetric_key2,
                b"invalid",
                &encapped_key2,
                &private_key2,
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_encapped_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair(b"key-id");
        let (_, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                &ciphertext,
                associated_data1,
                &encrypted_symmetric_key2,
                associated_data2,
                b"invalid",
                &private_key2,
            ),
            err(displays_as(contains_substring("failed to load encapped key")))
        );
    }

    #[test]
    fn test_decrypt_message_with_invalid_private_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair(b"key-id");
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair(b"key-id");
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert_that!(
            decrypt_message(
                &ciphertext,
                associated_data1,
                &encrypted_symmetric_key2,
                associated_data2,
                &encapped_key2,
                &private_key1, // Should be private_key2.
            ),
            err(displays_as(contains_substring("failed to unwrap symmetric key")))
        );
    }

    #[test]
    fn test_extract_key_from_cwt() {
        let (_, cose_key) = gen_keypair(b"key-id");
        let public_key = CoseSign1Builder::new()
            .payload(
                ClaimsSetBuilder::new()
                    .private_claim(
                        PUBLIC_KEY_CLAIM,
                        Value::from(cose_key.clone().to_vec().unwrap()),
                    )
                    .build()
                    .to_vec()
                    .unwrap(),
            )
            .build()
            .to_vec()
            .unwrap();
        assert_that!(extract_key_from_cwt(&public_key), ok(eq(cose_key)));
    }

    #[test]
    fn test_extract_key_from_cwt_with_invalid_cose_key() {
        let public_key = CoseSign1Builder::new()
            .payload(
                ClaimsSetBuilder::new()
                    .private_claim(PUBLIC_KEY_CLAIM, b"invalid".as_slice().into())
                    .build()
                    .to_vec()
                    .unwrap(),
            )
            .build()
            .to_vec()
            .unwrap();
        assert_that!(
            extract_key_from_cwt(&public_key),
            err(displays_as(contains_substring("failed to decode CoseKey")))
        );
    }

    #[test]
    fn test_extract_key_from_cwt_without_cose_key() {
        let public_key = CoseSign1Builder::new()
            .payload(ClaimsSetBuilder::new().build().to_vec().unwrap())
            .build()
            .to_vec()
            .unwrap();
        assert_that!(
            extract_key_from_cwt(&public_key),
            err(displays_as(contains_substring("missing public key claim")))
        );
    }

    #[test]
    fn test_extract_key_from_cwt_with_invalid_payload() {
        let public_key =
            CoseSign1Builder::new().payload(b"invalid".into()).build().to_vec().unwrap();
        assert_that!(
            extract_key_from_cwt(&public_key),
            err(displays_as(contains_substring("failed to decode CWT claims")))
        );
    }

    #[test]
    fn test_extract_key_from_cwt_with_invalid_cwt() {
        assert_that!(
            extract_key_from_cwt(b"invalid"),
            err(displays_as(contains_substring("failed to decode CWT claims")))
        );
    }
}
