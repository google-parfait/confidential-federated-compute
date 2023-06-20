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

use alloc::{format, vec};
use byteorder::{ByteOrder, LittleEndian};
use pipeline_transforms::{
    io::{EncryptionMode, RecordDecoder, RecordEncoder},
    proto::{PipelineTransform, TransformRequest, TransformResponse},
};

#[derive(Default)]
pub struct SumService {
    record_decoder: RecordDecoder,
    record_encoder: RecordEncoder,
}

impl PipelineTransform for SumService {
    fn transform(
        &mut self,
        request: &TransformRequest,
    ) -> Result<TransformResponse, micro_rpc::Status> {
        let mut sum: u64 = 0;
        for input in &request.inputs {
            let data = self.record_decoder.decode(input).map_err(|err| {
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
            sum = sum
                .checked_add(LittleEndian::read_u64(&data))
                .ok_or_else(|| {
                    micro_rpc::Status::new_with_message(
                        micro_rpc::StatusCode::InvalidArgument,
                        "addition overflow",
                    )
                })?;
        }

        let mut buffer = [0; 8];
        LittleEndian::write_u64(&mut buffer, sum);

        let output = self
            .record_encoder
            .encode(EncryptionMode::Unencrypted, &buffer)
            .map_err(|err| {
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
    fn test_transform_requires_8_bytes() {
        let mut service = SumService::default();
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
        let mut service = SumService::default();
        let request = TransformRequest {
            inputs: vec![encode_unencrypted(&[0, 0, 0, 0, 0, 0, 0, 0xFF]); 2],
            ..Default::default()
        };
        assert!(service.transform(&request).is_err());
    }

    #[test]
    fn test_transform_sums_inputs() -> Result<(), micro_rpc::Status> {
        let mut service = SumService::default();
        for count in 0..10 {
            let request = TransformRequest {
                inputs: (1..(count + 1))
                    .map(|i| encode_unencrypted(&[0, i, 0, 0, 0, 0, 0, 0]))
                    .collect(),
                ..Default::default()
            };
            let expected = encode_unencrypted(&[0, (1..(count + 1)).sum(), 0, 0, 0, 0, 0, 0]);
            assert_eq!(
                service.transform(&request)?,
                TransformResponse {
                    outputs: vec![expected],
                    ..Default::default()
                }
            );
        }
        Ok(())
    }
}
