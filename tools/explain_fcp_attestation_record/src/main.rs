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

//! A tool which takes a serialized `AttestationVerificationRecord` and prints a
//! human readable description of the ledger attestation evidence as well as the
//! data access policy contained within that record.

use std::io::{self, Read};
use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use federated_compute::proto::AttestationVerificationRecord;
use prost::Message;

#[derive(Parser, Debug)]
#[group(skip)]
struct Params {
    /// Path to the serialized AttestationVerificationRecord proto to inspect,
    /// or '-' to read from stdin.
    ///
    /// E.g. this can be one of the files output by a previous invocation of the
    /// "extract_attestation_records" tool from the FCP repository (see
    /// https://github.com/google-parfait/federated-compute/tree/main/fcp/client/attestation).
    #[arg(long, value_parser = parse_path_or_stdin)]
    pub record: PathOrStdin,
}

#[derive(Clone, Debug)]
enum PathOrStdin {
    Path(PathBuf),
    Stdin,
}

fn main() -> anyhow::Result<()> {
    let mut serialized_record: Vec<u8>;
    let params = Params::parse();
    match params.record {
        PathOrStdin::Path(record_path) => {
            println!("Inspecting record at {}.", record_path.display());
            serialized_record = fs::read(&record_path)
                .with_context(|| format!("failed to read record at {}", record_path.display()))?;
        }
        PathOrStdin::Stdin => {
            println!("Inspecting record provided via stdin.");
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
