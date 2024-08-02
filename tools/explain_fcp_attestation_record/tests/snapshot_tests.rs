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

//! Tests the 'explain_fcp_attestation_record' binary by running it over some
//! checked-in record data and comparing its output against snapshot files
//! containing the expected output. We use this testing approach because the
//! tool output generally quite long, and because we don't fully control the
//! format of it (the Oak Attestation Explanation library produces a good chunk
//! of the output), and hence it's easier to simply maintain and update 'golden'
//! outputs as needed.
//!
//! To make updating and reviewing changes to the snapshot files easier you can
//! use the `cargo insta review` command (installable using `cargo install
//! insta`). The test will generate updated snapshots with a '.new' filename
//! suffix whenever there's a test failure, however, and you can just manually
//! move those files into the original snapshot files, without the use of the
//! 'cargo insta' tool.

use std::io::Write;

use assert_cmd::Command;
use insta::assert_snapshot;
use prost::Message as _;

mod testdata;

/// Runs the explain tool over a record file that only contains valid ledger
/// attestation evidence and an empty access policy (which is not particularly
/// useful in practice, since it prevents any form of further data processing).
#[test]
fn test_explain_record_with_empty_data_access_policy_from_file() -> anyhow::Result<()> {
    // Create a temporary file containing the
    let mut record_file = tempfile::NamedTempFile::new()?;
    let serialized_record = testdata::record_with_empty_access_policy().encode_to_vec();
    record_file.write_all(&serialized_record)?;
    let record_file_path = record_file.path().to_str().unwrap();

    // Note: we run this test by actually running the command line binary. This
    // helps ensure that the binary actually works end-to-end, including
    // argument parsing etc. Subsequent tests forego this step and instead call
    // directly into the helper functions upon which the library is built.
    let cmd = Command::new(std::env::var("EXPLAIN_TOOL").unwrap())
        .arg("--record")
        .arg(record_file_path)
        .assert()
        .success();
    let output = std::str::from_utf8(&cmd.get_output().stdout)?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(output.replace(record_file_path, "{TMP_RECORD_FILE}"));
    Ok(())
}

/// Like [`test_explain_record_with_empty_data_access_policy_from_file`] but
/// passes the record via stdin.
#[test]
fn test_explain_record_with_empty_data_access_policy_from_stdin() -> anyhow::Result<()> {
    let serialized_record = testdata::record_with_empty_access_policy().encode_to_vec();

    // Note: we run this test by actually running the command line binary. This
    // helps ensure that the binary actually works end-to-end, including
    // argument parsing etc. Subsequent tests forego this step and instead call
    // directly into the helper functions upon which the library is built.
    let cmd = Command::new(std::env::var("EXPLAIN_TOOL").unwrap())
        .arg("--record")
        .arg("-")
        .write_stdin(serialized_record)
        .assert()
        .success();
    let output = std::str::from_utf8(&cmd.get_output().stdout)?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(output);
    Ok(())
}

/// Runs the explain tool over a record that contains valid ledger attestation
/// evidence and an access policy that is populated with a few transforms and
/// access budgets.
#[test]
fn test_explain_record_with_nonempty_data_access_policy() -> anyhow::Result<()> {
    // Note: we don't actually invoke the binary, and instead call the
    // explain_fcp_attestation_record library upon which the binary is built. This
    // makes it a bit easier to pass in records which we constructed/modified
    // within the test code, without having to write to temporary files etc.
    let mut buf = String::new();
    explain_fcp_attestation_record::explain_record(
        &mut buf,
        &testdata::record_with_nonempty_access_policy(),
    )?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(buf);
    Ok(())
}
