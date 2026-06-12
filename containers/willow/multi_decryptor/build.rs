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

use micro_rpc_build::{CompileOptions, ExternPath};
use std::io::Result;
use std::path::Path;

fn main() -> Result<()> {
    let protoc = std::env::var("PROTOC").expect("PROTOC must be set");
    std::env::set_var("PROTOC", protoc);

    let committee_selector_proto = std::env::var("COMMITTEE_SELECTOR_PROTO_FILE")
        .expect("COMMITTEE_SELECTOR_PROTO_FILE must be set");
    let reputable_decryptor_proto = std::env::var("REPUTABLE_DECRYPTOR_PROTO_FILE")
        .expect("REPUTABLE_DECRYPTOR_PROTO_FILE must be set");

    let cs_path = Path::new(&committee_selector_proto);
    let cs_dir = cs_path.parent().unwrap().parent().unwrap();

    let rd_path = Path::new(&reputable_decryptor_proto);
    let rd_dir = rd_path.parent().unwrap().parent().unwrap();

    let secure_aggregation_proto_file = std::env::var("SECURE_AGGREGATION_PROTO_PATH")
        .expect("SECURE_AGGREGATION_PROTO_PATH must be set");
    let suffix = "willow/proto/shell/ciphertexts.proto";
    let secure_aggregation_root = if secure_aggregation_proto_file.ends_with(suffix) {
        &secure_aggregation_proto_file[..secure_aggregation_proto_file.len() - suffix.len()]
    } else {
        panic!("unexpected format");
    };
    let secure_aggregation_dir = Path::new(secure_aggregation_root);

    let shell_encryption_proto_file = std::env::var("SHELL_ENCRYPTION_PROTO_PATH")
        .expect("SHELL_ENCRYPTION_PROTO_PATH must be set");
    let suffix_se = "shell_encryption/rns/rns_serialization.proto";
    let shell_encryption_root = if shell_encryption_proto_file.ends_with(suffix_se) {
        &shell_encryption_proto_file[..shell_encryption_proto_file.len() - suffix_se.len()]
    } else {
        panic!("unexpected format");
    };
    let shell_encryption_dir = Path::new(shell_encryption_root);

    micro_rpc_build::compile(
        &["proto/multi_decryptor.proto"],
        &[
            Path::new("proto"),
            cs_dir,
            rd_dir,
            secure_aggregation_dir,
            shell_encryption_dir,
        ],
        CompileOptions {
            extern_paths: vec![
                ExternPath::new(
                    ".apps.willow.committee_selector.service",
                    "::willow_committee_selector_service::apps::willow::committee_selector::service",
                ),
                ExternPath::new(
                    ".apps.willow.reputable_decryptor.service",
                    "::willow_reputable_decryptor_service::apps::willow::reputable_decryptor::service",
                ),
            ],
            bytes: vec![
                ".apps.willow.multi_decryptor.service.MultiDecryptorSnapshot".to_string(),
            ],
            ..Default::default()
        },
    );
    Ok(())
}
