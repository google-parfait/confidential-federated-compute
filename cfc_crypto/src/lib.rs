// Copyright 2023 The Confidential Federated Compute Authors.
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

use aes_gcm::{
    aead::{Aead, OsRng, Payload},
    Aes128Gcm, KeyInit,
};
use alloc::vec::Vec;
use anyhow::anyhow;
use hpke::{
    aead::AesGcm128, kdf::HkdfSha256, kem::X25519HkdfSha256, Deserializable, Kem, OpModeR, OpModeS,
    Serializable,
};

pub type PrivateKey = <X25519HkdfSha256 as Kem>::PrivateKey;

// A fixed random nonce, which is safe to reuse because the symmetric key is never reused.
static NONCE: [u8; 12] = [
    0x74, 0xDF, 0x8F, 0xD4, 0xBE, 0x34, 0xAF, 0x64, 0x7F, 0x5E, 0x54, 0xF6,
];

// The HPKE info field is not used.
static INFO: [u8; 0] = [];

/// Wraps a symmetric encryption key using HPKE.
///
/// # Return Value
///
/// Returns `Ok((encapped_key, encrypted_symmetric_key))` on success.
fn wrap_symmetric_key(
    symmetric_key: &[u8],
    recipient_public_key: &[u8],
    associated_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let public_key = <X25519HkdfSha256 as Kem>::PublicKey::from_bytes(recipient_public_key)
        .map_err(|err| anyhow!("failed to parse recipient public key: {:?}", err))?;
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
pub fn gen_keypair() -> (PrivateKey, Vec<u8>) {
    let (private_key, public_key) = <X25519HkdfSha256 as Kem>::gen_keypair(&mut OsRng);
    (private_key, public_key.to_bytes().to_vec())
}

/// Encrypts client data using a combination of HPKE and AEAD.
///
/// # Arguments
///
/// * `plaintext` - The message to be encrypted.
/// * `public_key` - The Curve 25519 SEC 1 encoded point public key of the recipient.
/// * `associated_data` - Additional data to be verified along with the message.
///
/// # Return Value
///
/// Returns `Ok((ciphertext, encapped_key, encrypted_symmetric_key))` on success.
pub fn encrypt_message(
    plaintext: &[u8],
    public_key: &[u8],
    associated_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    // Encrypt the plaintext using AEAD.
    let symmetric_key = Aes128Gcm::generate_key(OsRng);
    let cipher = Aes128Gcm::new(&symmetric_key);
    let ciphertext = cipher
        .encrypt(
            (&NONCE).into(),
            Payload {
                msg: plaintext,
                aad: associated_data,
            },
        )
        .map_err(|err| anyhow!("failed to encrypt plaintext: {:?}", err))?;

    // Encrypt the symmetric key using HPKE.
    let (encapped_key, encrypted_symmetric_key) =
        wrap_symmetric_key(symmetric_key.as_slice(), public_key, associated_data)?;
    Ok((ciphertext, encapped_key, encrypted_symmetric_key))
}

/// Unwraps and re-wraps the AEAD symmetric key for consumption by another party.
///
/// Instead of directly decrypting the message, this method allows the AEAD symmetric to be
/// passed off to another party without sharing the private key. In other words, it opens
/// the original HPKE message and then re-seals it for the new recipient.
///
/// # Arguments
///
/// * `encrypted_symmetric_key` - The encrypted symmetric key produced by `encrypt_message`.
/// * `serialized_encapped_key` - The encapped public key returned by `encrypt_message`.
/// * `private_key` - The corresponding private key for the public key passed to `encrypt_message`.
/// * `unwrap_associated_data` - The associated data passed to `encrypt_message`.
/// * `recipient_public_key` - The public key provided by the recipient's `CryptoContextGenerator`.
/// * `wrap_associated_data` - Additional data to be verified along with the message. This replaces
///    `unwrap_associated_data`.
///
/// # Return Value
///
/// Returns `Ok((encapped_key, encrypted_symmetric_key))` on success.
pub fn rewrap_symmetric_key(
    encrypted_symmetric_key: &[u8],
    serialized_encapped_key: &[u8],
    private_key: &PrivateKey,
    unwrap_associated_data: &[u8],
    recipient_public_key: &[u8],
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
/// * `ciphertext_associated_data` - The associated data passed to `encrypt_message`.
/// * `encrypted_symmetric_key` - The encrypted symmetric key produced by `rewrap_symmetric_key`.
/// * `encrypted_symmetric_key_associated_data` - The associated data passed to
///   `rewrap_symmetric_key`.
/// * `serialized_encapped_key` - The encapped public key returned by `rewrap_symmetric_key`.
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
    let cipher = Aes128Gcm::new_from_slice(&symmetric_key)
        .map_err(|err| anyhow!("failed to load symmetric key: {:?}", err))?;
    cipher
        .decrypt(
            (&NONCE).into(),
            Payload {
                msg: ciphertext,
                aad: ciphertext_associated_data,
            },
        )
        .map_err(|err| anyhow!("failed to decrypt data: {:?}", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_keypair_is_unique() {
        let (private_key1, public_key1) = gen_keypair();
        let (private_key2, public_key2) = gen_keypair();
        assert_ne!(private_key1.to_bytes(), private_key2.to_bytes());
        assert_ne!(public_key1, public_key2);
    }

    #[test]
    fn test_encrypt_rewrap_decrypt() -> anyhow::Result<()> {
        // Encrypt the original message.
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1)?;

        // Rewrap the symmetric key with a different key pair.
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
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
        assert!(encrypt_message(plaintext, b"invalid", associated_data).is_err());
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_encrypted_symmetric_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (_, encapped_key, _) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair();
        assert!(rewrap_symmetric_key(
            b"invalid",
            &encapped_key,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .is_err());
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_encapped_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (_, _, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair();
        assert!(rewrap_symmetric_key(
            &encrypted_symmetric_key,
            b"invalid",
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .is_err());
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_private_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (_, public_key1) = gen_keypair();
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        assert!(rewrap_symmetric_key(
            &encrypted_symmetric_key,
            &encapped_key,
            &private_key2, // Should be private_key1.
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .is_err());
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair();
        assert!(rewrap_symmetric_key(
            &encrypted_symmetric_key,
            &encapped_key,
            &private_key1,
            b"invalid",
            &public_key2,
            associated_data2,
        )
        .is_err());
    }

    #[test]
    fn test_rewrap_symmetric_key_with_invalid_public_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (_, encapped_key, encrypted_symmetric_key) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();

        let associated_data2 = b"associated data2";
        assert!(rewrap_symmetric_key(
            &encrypted_symmetric_key,
            &encapped_key,
            &private_key1,
            associated_data1,
            b"invalid",
            associated_data2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_ciphertext() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (_, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            b"invalid",
            associated_data1,
            &encrypted_symmetric_key2,
            associated_data2,
            &encapped_key2,
            &private_key2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_ciphertext_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            &ciphertext,
            b"invalid",
            &encrypted_symmetric_key2,
            associated_data2,
            &encapped_key2,
            &private_key2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_encrypted_symmetric_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        let (encapped_key2, _) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            &ciphertext,
            associated_data1,
            b"invalid",
            associated_data2,
            &encapped_key2,
            &private_key2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_encrypted_symmetric_key_associated_data() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            &ciphertext,
            associated_data1,
            &encrypted_symmetric_key2,
            b"invalid",
            &encapped_key2,
            &private_key2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_encapped_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (private_key2, public_key2) = gen_keypair();
        let (_, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            &ciphertext,
            associated_data1,
            &encrypted_symmetric_key2,
            associated_data2,
            b"invalid",
            &private_key2,
        )
        .is_err());
    }

    #[test]
    fn test_decrypt_message_with_invalid_private_key() {
        let plaintext = b"plaintext";
        let associated_data1 = b"associated data1";
        let (private_key1, public_key1) = gen_keypair();
        let (ciphertext, encapped_key1, encrypted_symmetric_key1) =
            encrypt_message(plaintext, &public_key1, associated_data1).unwrap();
        let associated_data2 = b"associated data2";
        let (_, public_key2) = gen_keypair();
        let (encapped_key2, encrypted_symmetric_key2) = rewrap_symmetric_key(
            &encrypted_symmetric_key1,
            &encapped_key1,
            &private_key1,
            associated_data1,
            &public_key2,
            associated_data2,
        )
        .unwrap();

        assert!(decrypt_message(
            &ciphertext,
            associated_data1,
            &encrypted_symmetric_key2,
            associated_data2,
            &encapped_key2,
            &private_key1, // Should be private_key2.
        )
        .is_err());
    }
}
