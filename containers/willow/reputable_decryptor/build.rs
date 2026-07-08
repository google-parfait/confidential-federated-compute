// Copyright 2026 The Trusted Computations Platform Authors.
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
use std::path::Path;

fn main() -> Result<()> {
    let secure_aggregation_proto_file = std::env::var("SECURE_AGGREGATION_PROTO_PATH")
        .expect("SECURE_AGGREGATION_PROTO_PATH must be set");

    let suffix = "willow/proto/shell/ciphertexts.proto";
    let secure_aggregation_root = if secure_aggregation_proto_file.ends_with(suffix) {
        &secure_aggregation_proto_file[..secure_aggregation_proto_file.len() - suffix.len()]
    } else {
        panic!(
            "SECURE_AGGREGATION_PROTO_PATH has unexpected format: {}",
            secure_aggregation_proto_file
        );
    };

    let secure_aggregation_dir = Path::new(secure_aggregation_root);

    let shell_encryption_proto_file = std::env::var("SHELL_ENCRYPTION_PROTO_PATH")
        .expect("SHELL_ENCRYPTION_PROTO_PATH must be set");

    let suffix_se = "shell_encryption/rns/rns_serialization.proto";
    let shell_encryption_root = if shell_encryption_proto_file.ends_with(suffix_se) {
        &shell_encryption_proto_file[..shell_encryption_proto_file.len() - suffix_se.len()]
    } else {
        panic!(
            "SHELL_ENCRYPTION_PROTO_PATH has unexpected format: {}",
            shell_encryption_proto_file
        );
    };

    let shell_encryption_dir = Path::new(shell_encryption_root);

    micro_rpc_build::compile(
        &["proto/willow_reputable_decryptor.proto"],
        &[Path::new("proto"), secure_aggregation_dir, shell_encryption_dir],
        micro_rpc_build::CompileOptions {
            bytes: vec![
                ".apps.willow.reputable_decryptor.service.ReputableDecryptorSnapshot".to_string(),
                ".apps.willow.reputable_decryptor.service.KeyStateSnapshot".to_string(),
            ],
            ..Default::default()
        },
    );
    Ok(())
}
