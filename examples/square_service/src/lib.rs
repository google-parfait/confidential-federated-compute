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
        record::{
            hpke_plus_aead_data::{RewrappedAssociatedData, SymmetricKeyAssociatedDataComponents},
            HpkePlusAeadData, Kind as RecordKind,
        },
        ConfigureAndAttestRequest, ConfigureAndAttestResponse, GenerateNoncesRequest,
        GenerateNoncesResponse, InitializeRequest, InitializeResponse, PipelineTransform,
        TransformRequest, TransformResponse,
    },
};

#[derive(Default)]
pub struct SquareService {
    record_decoder: Option<RecordDecoder>,
    record_encoder: RecordEncoder,
}

impl PipelineTransform for SquareService {
    fn initialize(
        &mut self,
        _request: InitializeRequest,
    ) -> Result<InitializeResponse, micro_rpc::Status> {
        Err(micro_rpc::Status::new(micro_rpc::StatusCode::Unimplemented))
    }

    fn configure_and_attest(
        &mut self,
        _request: ConfigureAndAttestRequest,
    ) -> Result<ConfigureAndAttestResponse, micro_rpc::Status> {
        self.record_decoder = Some(RecordDecoder::default());
        Ok(ConfigureAndAttestResponse {
            public_key: self.record_decoder.as_ref().unwrap().public_key().to_vec(),
            ..Default::default()
        })
    }

    fn generate_nonces(
        &mut self,
        request: GenerateNoncesRequest,
    ) -> Result<GenerateNoncesResponse, micro_rpc::Status> {
        let record_decoder = self.record_decoder.as_mut().ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "service has not been configured",
            )
        })?;
        let count: usize = request.nonces_count.try_into().map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("nonces_count is invalid: {:?}", err),
            )
        })?;
        Ok(GenerateNoncesResponse {
            nonces: record_decoder.generate_nonces(count),
        })
    }

    fn transform(
        &mut self,
        request: TransformRequest,
    ) -> Result<TransformResponse, micro_rpc::Status> {
        let record_decoder = self.record_decoder.as_mut().ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::FailedPrecondition,
                "service has not been configured",
            )
        })?;
        if request.inputs.len() != 1 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "transform expects exactly one input",
            ));
        }

        let data = record_decoder.decode(&request.inputs[0]).map_err(|err| {
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
                symmetric_key_associated_data_components:
                    Some(SymmetricKeyAssociatedDataComponents::RewrappedSymmetricKeyAssociatedData(
                        RewrappedAssociatedData {
                            reencryption_public_key: ref public_key,
                            ..
                        },
                    )),
                ..
            })) => EncryptionMode::HpkePlusAead {
                public_key,
                associated_data: header, // TODO(b/287284320): Update the header.
            },
            Some(RecordKind::HpkePlusAeadData(_)) => {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "rewrapped_symmetric_key_associated_data is required",
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
    fn test_initialize() {
        let mut service = SquareService::default();
        assert!(service.initialize(InitializeRequest::default()).is_err());
    }

    #[test]
    fn test_configure_and_attest() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let response = service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        assert!(!response.public_key.is_empty());
        Ok(())
    }

    #[test]
    fn test_generate_nonces() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        let response = service.generate_nonces(GenerateNoncesRequest { nonces_count: 3 })?;
        assert_eq!(response.nonces.len(), 3);
        assert_ne!(response.nonces[0], b"");
        assert_ne!(response.nonces[1], b"");
        assert_ne!(response.nonces[2], b"");
        Ok(())
    }

    #[test]
    fn test_generate_nonces_without_configure() {
        let mut service = SquareService::default();
        assert!(service
            .generate_nonces(GenerateNoncesRequest { nonces_count: 3 })
            .is_err());
    }

    #[test]
    fn test_generate_nonces_with_invalid_count() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        assert!(service
            .generate_nonces(GenerateNoncesRequest { nonces_count: -1 })
            .is_err());
        Ok(())
    }

    #[test]
    fn test_transform_without_configure() {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[1, 0, 0, 0, 0, 0, 0, 0]); 2],
            ..Default::default()
        };
        assert!(service.transform(request).is_err());
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
            assert!(service.transform(request).is_err());
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
            assert!(service.transform(request).is_err());
        }
    }

    #[test]
    fn test_transform_overflow() {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[0, 0, 0, 0, 0, 0, 0, 1])],
            ..Default::default()
        };
        assert!(service.transform(request).is_err());
    }

    #[test]
    fn test_transform_squares_input() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[2, 1, 0, 0, 0, 0, 0, 0])],
            ..Default::default()
        };
        assert_eq!(
            service.transform(request)?,
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
        let configure_response =
            service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        let nonces_response = service.generate_nonces(GenerateNoncesRequest { nonces_count: 1 })?;

        let (input, intermediary_private_key) = pipeline_transforms::io::create_rewrapped_record(
            &[4, 0, 0, 0, 0, 0, 0, 0],
            b"associated data",
            &configure_response.public_key,
            &nonces_response.nonces[0],
        )
        .unwrap();
        let request = TransformRequest {
            inputs: vec![input],
            ..Default::default()
        };
        let response = service.transform(request)?;
        assert_eq!(response.outputs.len(), 1);
        // The output record should be encrypted using the public key provided as the encrypted
        // symmetric key associated data.
        match response.outputs[0].kind {
            Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ref ciphertext,
                ref ciphertext_associated_data,
                ref encrypted_symmetric_key,
                ref encapsulated_public_key,
                ..
            })) => {
                assert_eq!(
                    cfc_crypto::decrypt_message(
                        ciphertext,
                        ciphertext_associated_data,
                        encrypted_symmetric_key,
                        ciphertext_associated_data,
                        encapsulated_public_key,
                        &intermediary_private_key
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
}
