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

use alloc::{boxed::Box, collections::BTreeMap, format, vec, vec::Vec};
use byteorder::{ByteOrder, LittleEndian};
use federated_compute::proto::BlobHeader;
use oak_crypto::signer::Signer;
use pipeline_transforms::{
    io::{DecryptionModeSet, EncryptionMode, RecordDecoder, RecordEncoder},
    proto::{
        record::{
            hpke_plus_aead_data::{RewrappedAssociatedData, SymmetricKeyAssociatedDataComponents},
            HpkePlusAeadData, Kind as RecordKind,
        },
        ConfigureAndAttestRequest, ConfigureAndAttestResponse, GenerateNoncesRequest,
        GenerateNoncesResponse, PipelineTransform, TransformRequest, TransformResponse,
    },
};
use prost::Message;
use prost_types::{value, Any, NullValue, Struct, Value};

pub struct SquareService {
    signer: Box<dyn Signer>,
    record_decoder: Option<RecordDecoder>,
    record_encoder: RecordEncoder,
    dest_node_id: Option<u32>,
}

impl SquareService {
    pub fn new(signer: Box<dyn Signer>) -> Self {
        Self { signer, record_decoder: None, record_encoder: RecordEncoder, dest_node_id: None }
    }

    /// Updates access_policy_node_id in a serialized BlobHeader.
    fn update_header(&self, header: &[u8]) -> Result<Vec<u8>, micro_rpc::Status> {
        let mut decoded = BlobHeader::decode(header).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                format!("failed to decode BlobHeader: {:?}", err),
            )
        })?;
        if let Some(id) = self.dest_node_id {
            decoded.access_policy_node_id = id;
        }
        Ok(decoded.encode_to_vec())
    }
}

impl PipelineTransform for SquareService {
    fn configure_and_attest(
        &mut self,
        request: ConfigureAndAttestRequest,
    ) -> Result<ConfigureAndAttestResponse, micro_rpc::Status> {
        // Read the destination node id from the configuration.
        self.dest_node_id = match request.configuration {
            Some(Any { type_url, value })
                if type_url == "type.googleapis.com/google.protobuf.UInt32Value" =>
            {
                Some(u32::decode(value.as_slice()).map_err(|err| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        format!("failed to unpack configuration as an UInt32Value: {:?}", err),
                    )
                })?)
            }
            Some(Any { type_url, .. }) if type_url.is_empty() => None,
            Some(Any { type_url, .. }) => {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    format!("unexpected type for configuration: {}", type_url),
                ));
            }
            None => None,
        };

        // Generate the configuration included in the public key claims.
        let config = Struct {
            fields: BTreeMap::from([(
                "dest".into(),
                Value {
                    kind: Some(match self.dest_node_id {
                        Some(id) => value::Kind::NumberValue(id as f64),
                        None => value::Kind::NullValue(NullValue::NullValue.into()),
                    }),
                },
            )]),
        };

        self.record_decoder = Some(
            RecordDecoder::create_with_config_and_modes(
                |msg| Ok(self.signer.sign(msg)),
                &config,
                DecryptionModeSet::all(),
            )
            .map_err(|err| {
                micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::Internal,
                    format!("failed to create RecordDecoder: {:?}", err),
                )
            })?,
        );
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
        Ok(GenerateNoncesResponse { nonces: record_decoder.generate_nonces(count) })
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
        let updated_header: Option<Vec<u8>>;
        let mode = match request.inputs[0].kind {
            Some(RecordKind::HpkePlusAeadData(HpkePlusAeadData {
                ciphertext_associated_data: ref header,
                symmetric_key_associated_data_components:
                    Some(SymmetricKeyAssociatedDataComponents::RewrappedSymmetricKeyAssociatedData(
                        RewrappedAssociatedData { reencryption_public_key: ref public_key, .. },
                    )),
                ..
            })) => {
                updated_header = Some(self.update_header(header)?);
                EncryptionMode::HpkePlusAead {
                    public_key,
                    associated_data: updated_header.as_ref().unwrap(),
                }
            }
            Some(RecordKind::HpkePlusAeadData(_)) => {
                return Err(micro_rpc::Status::new_with_message(
                    micro_rpc::StatusCode::InvalidArgument,
                    "rewrapped_symmetric_key_associated_data is required",
                ));
            }
            _ => EncryptionMode::Unencrypted,
        };

        let output = self.record_encoder.encode(mode, &buffer).map_err(|err| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::Internal,
                format!("failed to encode output: {:?}", err),
            )
        })?;
        Ok(TransformResponse { outputs: vec![output], num_ignored_inputs: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use cfc_crypto::CONFIG_PROPERTIES_CLAIM;
    use coset::{
        cwt::{ClaimName, ClaimsSet},
        CborSerializable, CoseSign1,
    };
    use oak_restricted_kernel_sdk::testing::MockSigner;
    use pipeline_transforms::proto::Record;
    use sha2::{Digest, Sha256};

    /// Helper function to create a SquareService.
    fn create_square_service() -> SquareService {
        SquareService::new(Box::new(MockSigner::create().unwrap()))
    }

    /// Helper function to convert data to an unencrypted Record.
    fn encode_unencrypted(data: &[u8]) -> Record {
        RecordEncoder::default().encode(EncryptionMode::Unencrypted, data).unwrap()
    }

    #[test]
    fn test_configure_and_attest() -> Result<(), micro_rpc::Status> {
        struct FakeSigner;
        impl Signer for FakeSigner {
            fn sign(&self, message: &[u8]) -> Vec<u8> {
                Sha256::digest(message).to_vec()
            }
        }

        let mut service = SquareService::new(Box::new(FakeSigner));
        let response = service.configure_and_attest(ConfigureAndAttestRequest {
            configuration: Some(Any {
                type_url: "type.googleapis.com/google.protobuf.UInt32Value".into(),
                value: 3u32.encode_to_vec(),
            }),
        })?;
        assert!(!response.public_key.is_empty());
        let cwt = CoseSign1::from_slice(&response.public_key).unwrap();
        cwt.verify_signature(b"", |signature, message| {
            anyhow::ensure!(signature == Sha256::digest(message).as_slice());
            Ok(())
        })
        .expect("signature mismatch");
        assert_eq!(
            ClaimsSet::from_slice(&cwt.payload.unwrap_or_default())
                .unwrap()
                .rest
                .iter()
                .find(|(name, _)| name == &ClaimName::PrivateUse(CONFIG_PROPERTIES_CLAIM))
                .map(|(_, value)| Struct::decode(value.as_bytes().unwrap().as_slice())),
            Some(Ok(Struct {
                fields: BTreeMap::from([(
                    "dest".into(),
                    Value { kind: Some(value::Kind::NumberValue(3.0)) }
                ),]),
            }))
        );
        Ok(())
    }

    #[test]
    fn test_configure_and_attest_with_empty_config() -> Result<(), micro_rpc::Status> {
        let mut service = create_square_service();

        // For backwards compatibility, configure_and_attest should succeed if the
        // configuration is empty (either unset or set to the default Any).
        service.configure_and_attest(ConfigureAndAttestRequest { configuration: None })?;
        service.configure_and_attest(ConfigureAndAttestRequest {
            configuration: Some(Any::default()),
        })?;
        Ok(())
    }

    #[test]
    fn test_configure_and_attest_with_invalid_config() {
        let mut service = create_square_service();

        assert!(service
            .configure_and_attest(ConfigureAndAttestRequest {
                configuration: Some(Any {
                    type_url: "type.googleapis.com/google.protobuf.DoubleValue".into(),
                    value: 3.0.encode_to_vec(),
                }),
            })
            .is_err());
    }

    #[test]
    fn test_generate_nonces() -> Result<(), micro_rpc::Status> {
        let mut service = create_square_service();
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
        let mut service = create_square_service();
        assert!(service.generate_nonces(GenerateNoncesRequest { nonces_count: 3 }).is_err());
    }

    #[test]
    fn test_generate_nonces_with_invalid_count() -> Result<(), micro_rpc::Status> {
        let mut service = create_square_service();
        service.configure_and_attest(ConfigureAndAttestRequest::default())?;
        assert!(service.generate_nonces(GenerateNoncesRequest { nonces_count: -1 }).is_err());
        Ok(())
    }

    #[test]
    fn test_transform_without_configure() {
        let mut service = create_square_service();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[1, 0, 0, 0, 0, 0, 0, 0]); 2],
            ..Default::default()
        };
        assert!(service.transform(request).is_err());
    }

    #[test]
    fn test_transform_requires_one_input() {
        let mut service = create_square_service();
        let input = encode_unencrypted(&[0; 8]);
        for count in [0, 2, 3] {
            let request =
                TransformRequest { inputs: vec![input.clone(); count], ..Default::default() };
            assert!(service.transform(request).is_err());
        }
    }

    #[test]
    fn test_transform_requires_8_bytes() {
        let mut service = create_square_service();
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
        let mut service = create_square_service();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[0, 0, 0, 0, 0, 0, 0, 1])],
            ..Default::default()
        };
        assert!(service.transform(request).is_err());
    }

    #[test]
    fn test_transform_squares_input() -> Result<(), micro_rpc::Status> {
        let mut service = create_square_service();
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
        let mut service = create_square_service();
        let configure_response = service.configure_and_attest(ConfigureAndAttestRequest {
            configuration: Some(Any {
                type_url: "type.googleapis.com/google.protobuf.UInt32Value".into(),
                value: 7u32.encode_to_vec(),
            }),
        })?;
        let nonces_response = service.generate_nonces(GenerateNoncesRequest { nonces_count: 1 })?;

        let header = BlobHeader {
            blob_id: b"blob-id".into(),
            key_id: b"key-id".into(),
            access_policy_node_id: 3,
            ..Default::default()
        };

        let (input, intermediary_private_key) = pipeline_transforms::io::create_rewrapped_record(
            &[4, 0, 0, 0, 0, 0, 0, 0],
            &header.encode_to_vec(),
            &configure_response.public_key,
            &nonces_response.nonces[0],
        )
        .unwrap();
        let request = TransformRequest { inputs: vec![input], ..Default::default() };
        let response = service.transform(request)?;
        assert_eq!(response.outputs.len(), 1);
        // The output record should be encrypted using the public key provided as the
        // encrypted symmetric key associated data.
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
                assert_eq!(
                    BlobHeader::decode(ciphertext_associated_data.as_slice()).unwrap(),
                    BlobHeader { access_policy_node_id: 7, ..header }
                );
            }
            _ => panic!("output is not encrypted"),
        };
        Ok(())
    }
}
