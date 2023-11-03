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

use crate::proto::{
    record::{HpkePlusAeadData, Kind as RecordKind},
    Record,
};
use alloc::vec::Vec;
use anyhow::anyhow;
use bitmask::bitmask;

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
}

impl RecordDecoder {
    /// Constructs a new RecordDecoder.
    pub fn new(allowed_modes: DecryptionModeSet) -> Self {
        let (private_key, public_key) = cfc_crypto::gen_keypair();
        Self {
            allowed_modes,
            private_key,
            public_key,
        }
    }

    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    /// Decodes an `Record` message to the corresponding message bytes.
    ///
    /// # Return Value
    ///
    /// Returns the unencrypted message on success.
    pub fn decode(&self, record: &Record) -> anyhow::Result<Vec<u8>> {
        match &record.kind {
            Some(RecordKind::UnencryptedData(d)) => {
                if !self.allowed_modes.contains(DecryptionMode::Unencrypted) {
                    return Err(anyhow!("Record.unencrypted_data is not supported"));
                }
                Ok(d.clone())
            }

            Some(RecordKind::HpkePlusAeadData(msg)) => {
                if !self.allowed_modes.contains(DecryptionMode::HpkePlusAead) {
                    return Err(anyhow!("Record.hpke_plus_aead_data is not supported"));
                }
                cfc_crypto::decrypt_message(
                    &msg.ciphertext,
                    &msg.ciphertext_associated_data,
                    &msg.encrypted_symmetric_key,
                    msg.encrypted_symmetric_key_associated_data
                        .as_ref()
                        .unwrap_or(&msg.ciphertext_associated_data),
                    &msg.encapsulated_public_key,
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
        Self::default()
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
                    ..Default::default()
                })
            }
        };
        Ok(Record { kind: Some(kind) })
    }
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

        let decoder = RecordDecoder::new(DecryptionMode::Unencrypted.into());
        assert_eq!(decoder.decode(&input)?, b"data".to_vec());
        Ok(())
    }

    #[test]
    fn test_decode_unencrypted_disallowed() {
        let input = Record {
            kind: Some(RecordKind::UnencryptedData(b"data".to_vec())),
            ..Default::default()
        };

        let decoder = RecordDecoder::new(DecryptionModeSet::none());
        assert!(decoder.decode(&input).is_err());
    }

    #[test]
    fn test_decode_hpke_plus_aead() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated data".to_vec();

        let decoder = RecordDecoder::new(DecryptionMode::HpkePlusAead.into());
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
        assert_eq!(decoder.decode(&input)?, plaintext.to_vec());
        Ok(())
    }

    #[test]
    fn test_decode_hpke_plus_aead_disallowed() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated_data".to_vec();

        let decoder = RecordDecoder::new(DecryptionModeSet::none());
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
    fn test_decode_hpke_plus_aead_with_decryption_error() -> anyhow::Result<()> {
        let plaintext = b"data";
        let ciphertext_associated_data = b"associated_data".to_vec();

        let decoder = RecordDecoder::new(DecryptionMode::HpkePlusAead.into());
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
        // encrypted_symmetric_key_associated_data should not be set.
        assert_eq!(msg.encrypted_symmetric_key_associated_data, None);
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
