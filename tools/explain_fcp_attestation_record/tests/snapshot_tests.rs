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

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use assert_cmd::Command;
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse as _,
};
use insta::assert_snapshot;
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use signed_endorsements_proto::fcp::confidentialcompute::signed_endorsements::PipelineConfiguration;
use tokio::task::JoinSet;
use verification_record_proto::fcp::confidentialcompute::AttestationVerificationRecord;
use verification_record_proto::payload_transparency_proto::fcp::confidentialcompute::signed_payload;

use testdata;

/// Runs the explain tool over a record file that only contains valid KMS
/// attestation evidence and an empty access policy (which is not particularly
/// useful in practice, since it prevents any form of further data processing).
#[tokio::test]
async fn test_explain_record_with_empty_data_access_policy_from_file() -> anyhow::Result<()> {
    let (record, access_policy, endorsed_evidence) = testdata::record_with_empty_access_policy();

    // Start a server to serve the access policy and endorsed evidence.
    let tmpdir = tempfile::tempdir()?;
    let socket = tmpdir.path().join("socket");
    let _handle = start_http_server(&socket, vec![access_policy, endorsed_evidence])?;

    // Create a temporary file containing the record.
    let mut record_file = tempfile::NamedTempFile::new()?;
    let serialized_record = record.encode_to_vec();
    record_file.write_all(&serialized_record)?;
    let record_file_path = record_file.path().to_str().unwrap().to_string();

    let cert_file = tmpdir.path().join("cert.pem");
    std::fs::write(&cert_file, include_bytes!("testdata/test_root.pem"))?;

    // Note: we run this test by actually running the command line binary. This
    // helps ensure that the binary actually works end-to-end, including
    // argument parsing etc. Subsequent tests forego this step and instead call
    // directly into the helper functions upon which the library is built.
    let record_file_path_clone = record_file_path.clone();
    let output = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let cmd = Command::new(std::env::var("EXPLAIN_TOOL").unwrap())
            .arg("--record")
            .arg(&record_file_path_clone)
            .arg("--test-unix-socket")
            .arg(&socket)
            .arg("--test-root-cert")
            .arg(&cert_file)
            .assert()
            .success();
        let stdout = std::str::from_utf8(&cmd.get_output().stdout)?;
        Ok(stdout.to_string())
    })
    .await??;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(output.replace(&record_file_path, "{TMP_RECORD_FILE}"));
    Ok(())
}

/// Like [`test_explain_record_with_empty_data_access_policy_from_file`] but
/// passes the record via stdin.
#[tokio::test]
async fn test_explain_record_with_empty_data_access_policy_from_stdin() -> anyhow::Result<()> {
    let (record, access_policy, endorsed_evidence) = testdata::record_with_empty_access_policy();

    // Start a server to serve the access policy and endorsed evidence.
    let tmpdir = tempfile::tempdir()?;
    let socket = tmpdir.path().join("socket");
    let _handle = start_http_server(&socket, vec![access_policy, endorsed_evidence])?;

    let serialized_record = record.encode_to_vec();

    let cert_file = tmpdir.path().join("cert.pem");
    std::fs::write(&cert_file, include_bytes!("testdata/test_root.pem"))?;

    // Note: we run this test by actually running the command line binary. This
    // helps ensure that the binary actually works end-to-end, including
    // argument parsing etc. Subsequent tests forego this step and instead call
    // directly into the helper functions upon which the library is built.
    let output = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let cmd = Command::new(std::env::var("EXPLAIN_TOOL").unwrap())
            .arg("--record")
            .arg("-")
            .arg("--test-unix-socket")
            .arg(&socket)
            .arg("--test-root-cert")
            .arg(&cert_file)
            .write_stdin(serialized_record)
            .assert()
            .success();
        let stdout = std::str::from_utf8(&cmd.get_output().stdout)?;
        Ok(stdout.to_string())
    })
    .await??;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(output);
    Ok(())
}

/// Runs the explain tool over a record that contains valid KMS attestation
/// evidence and an access policy that is populated with a few transforms and
/// access budgets.
#[tokio::test]
async fn test_explain_record_with_nonempty_data_access_policy() -> anyhow::Result<()> {
    let (record, access_policy, endorsed_evidence) = testdata::record_with_nonempty_access_policy();

    // Start a server to serve the access policy and endorsed evidence.
    let tmpdir = tempfile::tempdir()?;
    let socket = tmpdir.path().join("socket");
    let _handle = start_http_server(&socket, vec![access_policy, endorsed_evidence])?;

    // Note: we don't actually invoke the binary, and instead call the
    // explain_fcp_attestation_record library upon which the binary is built. This
    // makes it a bit easier to pass in records which we constructed/modified
    // within the test code, without having to write to temporary files etc.
    let mut buf = String::new();
    explain_fcp_attestation_record::explain_record(
        &mut buf,
        &record,
        &reqwest::Client::builder()
            .unix_socket(socket)
            .add_root_certificate(testdata::test_certs().0)
            .build()?,
    )
    .await?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(buf);
    Ok(())
}

/// Runs the explain tool over a legacy prebuilt record file that contains
/// unsupported ledger (Oak Restricted Kernel) attestation evidence without
/// signed_payload fields, ensuring a helpful error.
#[tokio::test]
async fn test_explain_record_with_ledger_evidence_without_signed_payload() -> anyhow::Result<()> {
    let raw_bytes =
        testdata::verification_record_with_ledger_evidence_without_signed_payload_bytes();
    let record = AttestationVerificationRecord::decode(raw_bytes)?;

    let mut buf = String::new();
    let result =
        explain_fcp_attestation_record::explain_record(&mut buf, &record, &reqwest::Client::new())
            .await;

    assert!(result.is_err());
    let err_msg = format!("{result:?}");
    assert!(
        err_msg.contains("Older attestation verification records may only be processable with older versions of this tool. Please try rerunning this tool at or before commit f5e8cfc6d3e20f3dcfe63ff8f8f11647ca2fd0f9."),
        "Unexpected error: {err_msg}",
    );
    Ok(())
}

/// Runs the explain tool over a legacy prebuilt record file that contains Oak
/// Containers (KMS) evidence directly in legacy inline fields without
/// signed_payload fields, ensuring a helpful error.
#[tokio::test]
async fn test_explain_record_with_kms_evidence_without_signed_payload() -> anyhow::Result<()> {
    let raw_bytes = testdata::verification_record_with_kms_evidence_without_signed_payload_bytes();
    let record = AttestationVerificationRecord::decode(raw_bytes)?;

    let mut buf = String::new();
    let result =
        explain_fcp_attestation_record::explain_record(&mut buf, &record, &reqwest::Client::new())
            .await;

    assert!(result.is_err());
    let err_msg = format!("{result:?}");
    assert!(
        err_msg.contains("Older attestation verification records may only be processable with older versions of this tool. Please try rerunning this tool at or before commit f5e8cfc6d3e20f3dcfe63ff8f8f11647ca2fd0f9."),
        "Unexpected error: {err_msg}",
    );
    Ok(())
}

/// Runs the explain tool over a SignedPayload signature structure for a valid
/// encryption key with KMS attestation evidence.
#[tokio::test]
async fn test_explain_signed_payload_with_encryption_key() -> anyhow::Result<()> {
    // Note: we don't actually invoke the binary, and instead call the
    // explain_fcp_attestation_record library upon which the binary is built. This
    // makes it a bit easier to pass in records which we constructed/modified
    // within the test code, without having to write to temporary files etc.
    let (_record, _access_policy, endorsed_evidence) = testdata::record_with_empty_access_policy();

    let headers = signed_payload::signature::Headers {
        claims: vec!["https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md".into()],
        oak_application_signature: Some(signed_payload::Signature {
            headers: signed_payload::signature::Headers {
                endorsed_evidence_sha256: Sha256::digest(&endorsed_evidence).to_vec(),
                ..Default::default()
            }
            .encode_to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();
    assert!(headers.len() < 128);
    let signature_structure =
        [b"\x0dSignedPayload".as_slice(), &[headers.len() as u8], &headers, b"\x00"].concat();

    // Start a server to serve the endorsed evidence.
    let tmpdir = tempfile::tempdir()?;
    let socket = tmpdir.path().join("socket");
    let _handle = start_http_server(&socket, vec![endorsed_evidence])?;

    let mut buf = String::new();
    explain_fcp_attestation_record::explain_signed_payload(
        &mut buf,
        &signature_structure,
        &reqwest::Client::builder()
            .unix_socket(socket)
            .add_root_certificate(testdata::test_certs().0)
            .build()?,
    )
    .await?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(buf);
    Ok(())
}

/// Runs the explain tool over a SignedPayload signature structure for a valid
/// PipelineConfiguration for a pipeline with an empty access policy.
#[tokio::test]
async fn test_explain_signed_payload_with_pipeline_configuration() -> anyhow::Result<()> {
    // Note: we don't actually invoke the binary, and instead call the
    // explain_fcp_attestation_record library upon which the binary is built. This
    // makes it a bit easier to pass in records which we constructed/modified
    // within the test code, without having to write to temporary files etc.
    let (_record, access_policy, _endorsed_evidence) = testdata::record_with_empty_access_policy();

    let pipeline_configuration =
        PipelineConfiguration { access_policy_sha256: Sha256::digest(&access_policy).to_vec() }
            .encode_to_vec();
    assert!(pipeline_configuration.len() < 128);
    let signature_structure = [
        b"\x0dSignedPayload".as_slice(),
        b"\x00",
        &[pipeline_configuration.len() as u8],
        &pipeline_configuration,
    ]
    .concat();

    // Start a server to serve the access policy.
    let tmpdir = tempfile::tempdir()?;
    let socket = tmpdir.path().join("socket");
    let _handle = start_http_server(&socket, vec![access_policy])?;

    let mut buf = String::new();
    explain_fcp_attestation_record::explain_signed_payload(
        &mut buf,
        &signature_structure,
        &reqwest::Client::builder()
            .unix_socket(socket)
            .add_root_certificate(testdata::test_certs().0)
            .build()?,
    )
    .await?;

    // Check that the expected output in the snapshot matches the actual output in
    // the buffer.
    assert_snapshot!(buf);
    Ok(())
}

/// Starts an HTTPS server that will serve the given files in response to GET
/// requests. Returns a handle to the running server.
fn start_http_server(
    path: &std::path::Path,
    files: Vec<Vec<u8>>,
) -> anyhow::Result<tokio::task::JoinSet<()>> {
    // Define a handler to serve the files.
    async fn handler(
        Path(sha256): Path<String>,
        State(files_by_sha256): State<Arc<HashMap<String, Vec<u8>>>>,
    ) -> axum::response::Response {
        match files_by_sha256.get(&sha256) {
            Some(data) => data.clone().into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }
    let files_by_sha256 =
        files.into_iter().map(|file| (hex::encode(Sha256::digest(&file)), file)).collect();
    let app = Router::new()
        .route("/data/transparency/sha2-256:{sha256}", axum::routing::get(handler))
        .with_state(Arc::new(files_by_sha256));

    // Load the TLS config.
    let rustls_config =
        axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(testdata::test_certs().1));

    // Bind to the Unix domain socket and start the server.
    let listener = std::os::unix::net::UnixListener::bind(path)?;
    listener.set_nonblocking(true).expect("couldn't set non-blocking");
    let serve = axum_server::tls_rustls::from_unix_rustls(listener, rustls_config)?
        .serve(app.into_make_service());
    let mut join_set = JoinSet::new();
    join_set.spawn(async move {
        serve.await.expect("server failed");
    });
    Ok(join_set)
}
