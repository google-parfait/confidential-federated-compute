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

extern crate alloc;

#[cfg(feature = "test")]
use crate::proto::record::hpke_plus_aead_data::RewrappedAssociatedData;
use crate::proto::{
    record::{
        hpke_plus_aead_data::{LedgerAssociatedData, SymmetricKeyAssociatedDataComponents},
        HpkePlusAeadData, Kind as RecordKind,
    },
    Record,
};
use alloc::{collections::BTreeSet, vec, vec::Vec};
use anyhow::anyhow;
use bitmask::bitmask;
use rand::{rngs::OsRng, RngCore};

bitmask! {
    /// The set of decryption modes supported by a RecordDecoder.
    pub mask DecryptionModeSet : u8 where
    /// A specific decryption mode, corresponding to an `Input.kind`.
    flags DecryptionMode {
        Unencrypted = 1,
        HpkePlusAead = 2,
    }
}

/// Decodes pipeline_transforms Record messages.
pub struct RecordDecoder {
    allowed_modes: DecryptionModeSet,
    private_key: cfc_crypto::PrivateKey,
    public_key: Vec<u8>,
    nonces: BTreeSet<Vec<u8>>,
}

impl RecordDecoder {
    /// Constructs a new RecordDecoder.
    pub fn new(allowed_modes: DecryptionModeSet) -> Self {
        let (private_key, public_key) = cfc_crypto::gen_keypair();
        Self {
            allowed_modes,
            private_key,
            public_key,
            nonces: BTreeSet::new(),
        }
    }

    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    /// Generates new nonces that can be used when decoding messages.
    pub fn generate_nonces(&mut self, count: usize) -> Vec<Vec<u8>> {
        let mut nonces = Vec::with_capacity(count);
        while nonces.len() < count {
            let mut nonce = vec![0u8; 8];
            OsRng.fill_bytes(nonce.as_mut_slice());
            if self.nonces.insert(nonce.clone()) {
                nonces.push(nonce);
            }
        }
        nonces
    }

    /// Decodes an `Record` message to the corresponding message bytes.
    ///
    /// # Return Value
    ///
    /// Returns the unencrypted message on success.
    pub fn decode(&mut self, record: &Record) -> anyhow::Result<Vec<u8>> {
        match &record.kind {
            Some(RecordKind::UnencryptedData(d)) => {
                if !self.allowed_modes.contains(DecryptionMode::Unencrypted) {
                    return Err(anyhow!("Record.unencrypted_data is not supported"));
                }
                Ok(d.clone())
            }

            Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ref ciphertext,
                ref ciphertext_associated_data,
                ref encrypted_symmetric_key,
                symmetric_key_associated_data_components:
                    Some(SymmetricKeyAssociatedDataComponents::RewrappedSymmetricKeyAssociatedData(
                        ref ad,
                    )),
                ref encapsulated_public_key,
                ..
            })) => {
                if !self.allowed_modes.contains(DecryptionMode::HpkePlusAead) {
                    return Err(anyhow!("Record.hpke_plus_aead_data is not supported"));
                }
                if !self.nonces.remove(&ad.nonce) {
                    return Err(anyhow!("invalid nonce"));
                }
                cfc_crypto::decrypt_message(
                    ciphertext,
                    ciphertext_associated_data,
                    encrypted_symmetric_key,
                    &[ad.reencryption_public_key.as_slice(), ad.nonce.as_slice()].concat(),
                    encapsulated_public_key,
                    &self.private_key,
                )
            }

            _ => Err(anyhow!("unsupported Record kind")),
        }
    }
}

impl Default for RecordDecoder {
    /// Creates an `RecordDecoder` that supports all decryption modes.
    fn default() -> Self {
        Self::new(DecryptionModeSet::all())
    }
}

/// The type of encryption that should be performed by a RecordEncoder.
#[derive(Clone, Copy)]
pub enum EncryptionMode<'a> {
    Unencrypted,
    HpkePlusAead {
        public_key: &'a [u8],
        associated_data: &'a [u8],
    },
}

/// Encodes pipeline_transforms Record messages.
#[derive(Default)]
pub struct RecordEncoder;

impl RecordEncoder {
    /// Constructs a new `RecordEncoder`.
    pub fn new() -> Self {
        Self
    }

    /// Encodes data as a `Record` message.
    pub fn encode(&self, mode: EncryptionMode, data: &[u8]) -> anyhow::Result<Record> {
        let kind = match mode {
            EncryptionMode::Unencrypted => RecordKind::UnencryptedData(data.to_vec()),

            EncryptionMode::HpkePlusAead {
                public_key,
                associated_data,
            } => {
                let (ciphertext, encapsulated_public_key, encrypted_symmetric_key) =
                    cfc_crypto::encrypt_message(data, public_key, associated_data)?;
                RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                    ciphertext,
                    ciphertext_associated_data: associated_data.to_vec(),
                    encrypted_symmetric_key,
                    encapsulated_public_key,
                    symmetric_key_associated_data_components: Some(
                        SymmetricKeyAssociatedDataComponents::LedgerSymmetricKeyAssociatedData(
                            LedgerAssociatedData {
                                record_header: associated_data.to_vec(),
                            },
                        ),
                    ),
                    ..Default::default()
                })
            }
        };
        Ok(Record { kind: Some(kind) })
    }
}

/// Creates a Record that has been rewrapped so that it can be decrypted by a RecordDecoder.
#[cfg(feature = "test")]
pub fn create_rewrapped_record(
    plaintext: &[u8],
    associated_data: &[u8],
    recipient_public_key: &[u8],
    nonce: &[u8],
) -> anyhow::Result<(Record, cfc_crypto::PrivateKey)> {
    let (intermediary_private_key, intermediary_public_key) = cfc_crypto::gen_keypair();
    let (ciphertext, encapsulated_public_key, encrypted_symmetric_key) =
        cfc_crypto::encrypt_message(plaintext, &intermediary_public_key, associated_data)?;

    let (encapsulated_public_key, encrypted_symmetric_key) = cfc_crypto::rewrap_symmetric_key(
        &encrypted_symmetric_key,
        &encapsulated_public_key,
        &intermediary_private_key,
        &associated_data,
        recipient_public_key,
        &[intermediary_public_key.as_slice(), nonce].concat(),
    )?;

    Ok((
        Record {
            kind: Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ciphertext,
                ciphertext_associated_data: associated_data.to_vec(),
                symmetric_key_associated_data_components: Some(
                    SymmetricKeyAssociatedDataComponents::RewrappedSymmetricKeyAssociatedData(
                        RewrappedAssociatedData {
                            reencryption_public_key: intermediary_public_key,
                            nonce: nonce.to_vec(),
                        },
                    ),
                ),
                encrypted_symmetric_key,
                encapsulated_public_key,
                ..Default::default()
            })),
            ..Default::default()
        },
        intermediary_private_key,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_unencrypted() -> anyhow::Result<()> {
        let input = Record {
            kind: Some(RecordKind::UnencryptedData(b"data".to_vec())),
            ..Default::default()
        };

        let mut decoder = RecordDecoder::new(DecryptionMode::Unencrypted.into());
        assert_eq!(decoder.decode(&input)?, b"data".to_vec());
        Ok(())
    }

    #[test]
    fn test_decode_unencrypted_disallowed() {
        let input = Record {
            kind: Some(RecordKind::UnencryptedData(b"data".to_vec())),
            ..Default::default()
        };

        let mut decoder = RecordDecoder::new(DecryptionModeSet::none());
        assert!(decoder.decode(&input).is_err());
    }

    #[test]
    fn test_decode_hpke_plus_aead() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated data";

        let mut decoder = RecordDecoder::new(DecryptionMode::HpkePlusAead.into());
        let nonce = &decoder.generate_nonces(1)[0];
        let (input, _) = create_rewrapped_record(
            plaintext,
            ciphertext_associated_data,
            decoder.public_key(),
            nonce,
        )?;
        assert_eq!(decoder.decode(&input)?, plaintext.to_vec());
        Ok(())
    }

    #[test]
    fn test_decode_hpke_plus_aead_disallowed() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated_data".to_vec();

        let mut decoder = RecordDecoder::new(DecryptionModeSet::none());
        let (ciphertext, encapsulated_public_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(
                plaintext,
                decoder.public_key(),
                &ciphertext_associated_data,
            )?;
        let input = Record {
            kind: Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ciphertext,
                ciphertext_associated_data,
                encrypted_symmetric_key,
                encapsulated_public_key,
                ..Default::default()
            })),
            ..Default::default()
        };
        assert!(decoder.decode(&input).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_hpke_plus_aead_with_invalid_nonce() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated data";

        let mut decoder = RecordDecoder::new(DecryptionMode::HpkePlusAead.into());
        let (input, _) = create_rewrapped_record(
            plaintext,
            ciphertext_associated_data,
            decoder.public_key(),
            b"nonce", // Use a nonce that wasn't generated by the RecordDecoder.
        )?;
        assert!(decoder.decode(&input).is_err());
        Ok(())
    }

    #[test]
    fn test_decode_hpke_plus_aead_with_decryption_error() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated_data".to_vec();

        let mut decoder = RecordDecoder::new(DecryptionMode::HpkePlusAead.into());
        let (ciphertext, encapsulated_public_key, encrypted_symmetric_key) =
            cfc_crypto::encrypt_message(
                plaintext,
                decoder.public_key(),
                &ciphertext_associated_data,
            )?;
        let input = Record {
            kind: Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ciphertext,
                ciphertext_associated_data: b"wrong associated data".to_vec(),
                encrypted_symmetric_key,
                encapsulated_public_key,
                ..Default::default()
            })),
            ..Default::default()
        };
        assert!(decoder.decode(&input).is_err());
        Ok(())
    }

    #[test]
    fn test_encode_unencrypted() -> anyhow::Result<()> {
        let data = b"data";
        assert_eq!(
            RecordEncoder::default().encode(EncryptionMode::Unencrypted, data)?,
            Record {
                kind: Some(RecordKind::UnencryptedData(data.to_vec())),
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_encode_hpke_plus_aead() -> anyhow::Result<()> {
        let plaintext = b"data";
        let associated_data = b"associated data";
        let (private_key, public_key) = cfc_crypto::gen_keypair();
        let output = RecordEncoder::default().encode(
            EncryptionMode::HpkePlusAead {
                public_key: &public_key,
                associated_data,
            },
            plaintext,
        )?;
        let msg = match output.kind {
            Some(RecordKind::HpkePlusAeadData(msg)) => msg,
            _ => return Err(anyhow!("expected HpkePlusAeadData")),
        };

        assert_eq!(&msg.ciphertext_associated_data, associated_data);
        assert_eq!(
            msg.symmetric_key_associated_data_components,
            Some(
                SymmetricKeyAssociatedDataComponents::LedgerSymmetricKeyAssociatedData(
                    LedgerAssociatedData {
                        record_header: associated_data.to_vec(),
                    }
                )
            )
        );
        assert_eq!(
            cfc_crypto::decrypt_message(
                &msg.ciphertext,
                &msg.ciphertext_associated_data,
                &msg.encrypted_symmetric_key,
                &msg.ciphertext_associated_data,
                &msg.encapsulated_public_key,
                &private_key
            )?,
            plaintext
        );
        Ok(())
    }

    #[test]
    fn test_encode_hpke_plus_aead_with_encryption_error() {
        let plaintext = b"data";
        let associated_data = b"associated data";
        assert!(RecordEncoder::default()
            .encode(
                EncryptionMode::HpkePlusAead {
                    public_key: b"invalid key",
                    associated_data,
                },
                plaintext,
            )
            .is_err());
    }
}
