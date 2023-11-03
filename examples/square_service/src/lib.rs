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

use alloc::{format, vec};
use byteorder::{ByteOrder, LittleEndian};
use pipeline_transforms::{
    io::{EncryptionMode, RecordDecoder, RecordEncoder},
    proto::{
        record::HpkePlusAeadData, record::Kind as RecordKind, ConfigureAndAttestRequest,
        ConfigureAndAttestResponse, GenerateNoncesRequest, GenerateNoncesResponse,
        InitializeRequest, InitializeResponse, PipelineTransform, TransformRequest,
        TransformResponse,
    },
};

#[derive(Default)]
pub struct SquareService {
    record_decoder: RecordDecoder,
    record_encoder: RecordEncoder,
}

impl PipelineTransform for SquareService {
    fn initialize(
        &mut self,
        _request: &InitializeRequest,
    ) -> Result<InitializeResponse, micro_rpc::Status> {
        Ok(InitializeResponse {
            public_key: self.record_decoder.public_key().to_vec(),
            ..Default::default()
        })
    }

    fn configure_and_attest(
        &mut self,
        _request: &ConfigureAndAttestRequest,
    ) -> Result<ConfigureAndAttestResponse, micro_rpc::Status> {
        Err(micro_rpc::Status::new(micro_rpc::StatusCode::Unimplemented))
    }

    fn generate_nonces(
        &mut self,
        _request: &GenerateNoncesRequest,
    ) -> Result<GenerateNoncesResponse, micro_rpc::Status> {
        Err(micro_rpc::Status::new(micro_rpc::StatusCode::Unimplemented))
    }

    fn transform(
        &mut self,
        request: &TransformRequest,
    ) -> Result<TransformResponse, micro_rpc::Status> {
        if request.inputs.len() != 1 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "transform expects exactly one input",
            ));
        }

        let data = self
            .record_decoder
            .decode(&request.inputs[0])
            .map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("failed to decode input: {:?}", err),
                )
            })?;
        if data.len() != 8 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "input must be 8 bytes",
            ));
        }

        let value = LittleEndian::read_u64(&data);
        let product = value.checked_mul(value).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "multiplication overflow",
            )
        })?;
        let mut buffer = [0; 8];
        LittleEndian::write_u64(&mut buffer, product);

        // SquareService maintains the encryption of its input.
        let mode = match request.inputs[0].kind {
            Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ciphertext_associated_data: ref header,
                encrypted_symmetric_key_associated_data: Some(ref public_key),
                ..
            })) => EncryptionMode::HpkePlusAead {
                public_key,
                associated_data: header, // TODO(b/287284320): Update the header.
            },
            Some(RecordKind::HpkePlusAeadData(_)) => {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "encrypted_symmetric_key_associated_data is required",
                ))
            }
            _ => EncryptionMode::Unencrypted,
        };

        let output = self.record_encoder.encode(mode, &buffer).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                format!("failed to encode output: {:?}", err),
            )
        })?;
        Ok(TransformResponse {
            outputs: vec![output],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipeline_transforms::proto::Record;

    /// Helper function to convert data to an unencrypted Record.
    fn encode_unencrypted(data: &[u8]) -> Record {
        RecordEncoder::default()
            .encode(EncryptionMode::Unencrypted, data)
            .unwrap()
    }

    #[test]
    fn test_initialize() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let response = service.initialize(&InitializeRequest::default())?;
        assert_ne!(response.public_key, vec!());
        Ok(())
    }

    #[test]
    fn test_transform_requires_one_input() {
        let mut service = SquareService::default();
        let input = encode_unencrypted(&[0; 8]);
        for count in [0, 2, 3] {
            let request = TransformRequest {
                inputs: vec![input.clone(); count],
                ..Default::default()
            };
            assert!(service.transform(&request).is_err());
        }
    }

    #[test]
    fn test_transform_requires_8_bytes() {
        let mut service = SquareService::default();
        for length in (0..7).chain(9..16) {
            let request = TransformRequest {
                inputs: vec![encode_unencrypted(&vec![0; length])],
                ..Default::default()
            };
            assert!(service.transform(&request).is_err());
        }
    }

    #[test]
    fn test_transform_overflow() {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[0, 0, 0, 0, 0, 0, 0, 1])],
            ..Default::default()
        };
        assert!(service.transform(&request).is_err());
    }

    #[test]
    fn test_transform_squares_input() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[2, 1, 0, 0, 0, 0, 0, 0])],
            ..Default::default()
        };
        assert_eq!(
            service.transform(&request)?,
            TransformResponse {
                outputs: vec![encode_unencrypted(&[4, 4, 1, 0, 0, 0, 0, 0])],
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_transform_encrypted() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let initialize_response = service.initialize(&InitializeRequest::default())?;

        // Emulate the Ledger rewrapping the record.
        let (ledger_private_key, ledger_public_key) = cfc_crypto::gen_keypair();
        let associated_data = b"associated data";
        let initial_record = RecordEncoder::default()
            .encode(
                EncryptionMode::HpkePlusAead {
                    public_key: &ledger_public_key,
                    associated_data,
                },
                &[4, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap();
        let initial_hpke_plus_aead_data = match initial_record.kind {
            Some(RecordKind::HpkePlusAeadData(data)) => data,
            _ => panic!("record is not encrypted"),
        };
        let (encapsulated_public_key, encrypted_symmetric_key) = cfc_crypto::rewrap_symmetric_key(
            &initial_hpke_plus_aead_data.encrypted_symmetric_key,
            &initial_hpke_plus_aead_data.encapsulated_public_key,
            &ledger_private_key,
            associated_data,
            &initialize_response.public_key,
            /* wrap_associated_data=*/ &ledger_public_key,
        )
        .unwrap();
        let rewrapped_record = Record {
            kind: Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                encrypted_symmetric_key,
                encapsulated_public_key,
                encrypted_symmetric_key_associated_data: Some(ledger_public_key),
                ..initial_hpke_plus_aead_data
            })),
            ..Default::default()
        };

        // Send the transform request.
        let request = TransformRequest {
            inputs: vec![rewrapped_record],
            ..Default::default()
        };
        let response = service.transform(&request)?;
        assert_eq!(response.outputs.len(), 1);
        // The output record should be encrypted using the public key provided as the encrypted
        // symmetric key associated data.
        match response.outputs[0].kind {
            Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ref ciphertext,
                ref ciphertext_associated_data,
                ref encrypted_symmetric_key,
                ref encrypted_symmetric_key_associated_data,
                ref encapsulated_public_key,
                ..
            })) => {
                assert!(encrypted_symmetric_key_associated_data.is_none());
                assert_eq!(
                    cfc_crypto::decrypt_message(
                        ciphertext,
                        ciphertext_associated_data,
                        encrypted_symmetric_key,
                        ciphertext_associated_data,
                        encapsulated_public_key,
                        &ledger_private_key
                    )
                    .as_ref()
                    .expect("failed to decrypt output record"),
                    &[16, 0, 0, 0, 0, 0, 0, 0]
                );
            }
            _ => panic!("output is not encrypted"),
        };
        Ok(())
    }

    #[test]
    fn test_transform_encrypted_without_encrypted_symmetric_key_associated_data(
    ) -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let initialize_response = service.initialize(&InitializeRequest::default())?;
        let mode = EncryptionMode::HpkePlusAead {
            public_key: &initialize_response.public_key,
            associated_data: b"associated data",
        };

        let record_encoder = RecordEncoder::default();
        let request = TransformRequest {
            inputs: vec![record_encoder
                .encode(mode, &[4, 0, 0, 0, 0, 0, 0, 0])
                .unwrap()],
            ..Default::default()
        };
        assert!(service.transform(&request).is_err());
        Ok(())
    }
}
