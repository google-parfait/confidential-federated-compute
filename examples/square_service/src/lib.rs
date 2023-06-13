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

use alloc::vec;
use byteorder::{ByteOrder, LittleEndian};
use pipeline_transforms::proto::{
    transform_response::Output, PipelineTransform, TransformRequest, TransformResponse,
};

#[derive(Default)]
pub struct SquareService;

impl PipelineTransform for SquareService {
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

        if request.inputs[0].data.len() != 8 {
            return Err(micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "input must be 8 bytes",
            ));
        }

        let value = LittleEndian::read_u64(&request.inputs[0].data);
        let product = value.checked_mul(value).ok_or_else(|| {
            micro_rpc::Status::new_with_message(
                micro_rpc::StatusCode::InvalidArgument,
                "multiplication overflow",
            )
        })?;
        let mut buffer = [0; 8];
        LittleEndian::write_u64(&mut buffer, product);

        Ok(TransformResponse {
            outputs: vec![Output {
                data: buffer.to_vec(),
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipeline_transforms::proto::transform_request::Input;

    #[test]
    fn test_transform_requires_one_input() {
        let mut service = SquareService::default();
        let input = Input {
            data: vec![0; 8],
            ..Default::default()
        };
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
                inputs: vec![Input {
                    data: vec![0; length],
                    ..Default::default()
                }],
                ..Default::default()
            };
            assert!(service.transform(&request).is_err());
        }
    }

    #[test]
    fn test_transform_overflow() {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![Input {
                data: vec![0, 0, 0, 0, 0, 0, 0, 1],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(service.transform(&request).is_err());
    }

    #[test]
    fn test_transform_squares_input() -> Result<(), micro_rpc::Status> {
        let mut service = SquareService::default();
        let request = TransformRequest {
            inputs: vec![Input {
                data: vec![2, 1, 0, 0, 0, 0, 0, 0],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            service.transform(&request)?,
            TransformResponse {
                outputs: vec![Output {
                    data: vec![4, 4, 1, 0, 0, 0, 0, 0],
                    ..Default::default()
                }],
                ..Default::default()
            }
        );
        Ok(())
    }
}
