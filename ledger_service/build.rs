// Copyright 2024 Google LLC.
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

fn main() {
    micro_rpc_build::compile(
        &["proto/replication.proto"],
        &[
            "proto",
            std::env::var("EVIDENCE_PROTO")
                .unwrap()
                .strip_suffix("/proto/attestation/evidence.proto")
                .unwrap(),
            std::env::var("LEDGER_PROTO")
                .unwrap()
                .strip_suffix("/fcp/protos/confidentialcompute/ledger.proto")
                .unwrap(),
            std::env::var("DESCRIPTOR_PROTO")
                .unwrap()
                .strip_suffix("/google/protobuf/descriptor.proto")
                .unwrap(),
        ],
        micro_rpc_build::CompileOptions {
            extern_paths: vec![
                micro_rpc_build::ExternPath::new(
                    ".oak.attestation.v1",
                    "::oak_proto_rust::oak::attestation::v1",
                ),
                micro_rpc_build::ExternPath::new(
                    ".fcp.confidentialcompute",
                    "::federated_compute::proto",
                ),
            ],
            ..Default::default()
        },
    );
}
