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

//! A tool which takes a serialized `AttestationVerificationRecord` or
//! `DataAccessPolicy` and prints a human readable description.

use std::io::{self, Read};
use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use prost::Message;
use verification_record_proto::{
    access_policy_proto::fcp::confidentialcompute::DataAccessPolicy,
    fcp::confidentialcompute::AttestationVerificationRecord,
};

#[derive(Parser, Debug)]
#[group(required = true, multiple = false)]
struct Params {
    /// Path to the serialized AttestationVerificationRecord proto to inspect,
    /// or '-' to read from stdin.
    ///
    /// E.g. this can be one of the files output by a previous invocation of the
    /// "extract_attestation_records" tool from the FCP repository (see
    /// https://github.com/google-parfait/federated-compute/tree/main/fcp/client/attestation).
    #[arg(long, value_parser = parse_path_or_stdin, conflicts_with = "access_policy")]
    pub record: Option<PathOrStdin>,

    /// Path to the serialized DataAccessPolicy proto to inspect, or '-' to
    /// read from stdin.
    ///
    /// E.g. this can be the payload of an access policy Oak endorsement.
    /// (see
    /// https://github.com/google-parfait/confidential-federated-compute/inspecting_endorsements).
    #[arg(long, value_parser = parse_path_or_stdin)]
    pub access_policy: Option<PathOrStdin>,
}

#[derive(Clone, Debug)]
enum PathOrStdin {
    Path(PathBuf),
    Stdin,
}

fn main() -> anyhow::Result<()> {
    let params = Params::parse();
    match params {
        Params { record: Some(record_param), .. } => {
            let mut serialized_record: Vec<u8>;
            match record_param {
                PathOrStdin::Path(record_path) => {
                    println!(
                        "Inspecting AttestationVerificationRecord at {}.",
                        record_path.display()
                    );
                    serialized_record = fs::read(&record_path).with_context(|| {
                        format!("failed to read record at {}", record_path.display())
                    })?;
                }
                PathOrStdin::Stdin => {
                    println!("Inspecting AttestationVerificationRecord provided via stdin.");
                    serialized_record = Vec::new();
                    io::stdin().read_to_end(&mut serialized_record)?;
                }
            }
            println!();

            let record = AttestationVerificationRecord::decode(serialized_record.as_slice())
                .context("failed to parse record")?;

            let mut explanation = String::new();
            explain_fcp_attestation_record::explain_record(&mut explanation, &record)
                .context("failed to explain record")?;
            println!("{}", explanation);
        }
        Params { access_policy: Some(access_policy_param), .. } => {
            let mut serialized_policy: Vec<u8>;
            match access_policy_param {
                PathOrStdin::Path(policy_path) => {
                    println!("Inspecting DataAccessPolicy at {}.", policy_path.display());
                    serialized_policy = fs::read(&policy_path).with_context(|| {
                        format!("failed to read record at {}", policy_path.display())
                    })?;
                }
                PathOrStdin::Stdin => {
                    println!("Inspecting DataAccessPolicy provided via stdin.");
                    serialized_policy = Vec::new();
                    io::stdin().read_to_end(&mut serialized_policy)?;
                }
            }
            println!();

            let policy = DataAccessPolicy::decode(serialized_policy.as_slice())
                .context("failed to parse policy")?;

            let mut explanation = String::new();
            explain_fcp_attestation_record::explain_data_access_policy(&mut explanation, &policy)
                .context("failed to explain policy")?;
            println!("{}", explanation);
        }
        _ => unreachable!(),
    }

    Ok(())
}

/// Parses the record argument, which can be either a valid path or '-' to
/// indicate reading from stdin.
fn parse_path_or_stdin(s: &str) -> Result<PathOrStdin, String> {
    if s == "-" {
        Ok(PathOrStdin::Stdin)
    } else if !fs::metadata(s).map_err(|err| err.to_string())?.is_file() {
        Err(format!("path {s} does not represent a file"))
    } else {
        Ok(PathOrStdin::Path(PathBuf::from(s)))
    }
}
