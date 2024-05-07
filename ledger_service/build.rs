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

use std::io::Result;

fn main() -> Result<()> {
    micro_rpc_build::compile(
        &["proto/replication.proto"],
        &[
            "proto",
            "../third_party/federated_compute/federated-compute",
            "../third_party/federated_compute/rust_proto_stubs",
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
    Ok(())
}
