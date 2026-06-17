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

use anyhow::{Context, bail, ensure};
use fed_sql_container_config_proto::fcp::confidentialcompute::FedSqlContainerConfigConstraints;
use futures_util::TryFutureExt as _;
use integer_encoding::VarIntReader;
use messages_proto::oak::session::v1::EndorsedEvidence;
use oak_attestation_explain::{HumanReadableExplanation, HumanReadableTitle};
use oak_proto_rust::oak::attestation::v1::{
    Evidence, OakContainersData, OakRestrictedKernelData, ReferenceValues,
    extracted_evidence::EvidenceValues,
};
use prost::Message;
use prost_types::Any;
use sha2::{Digest, Sha256};
use signed_endorsements_proto::fcp::confidentialcompute::signed_endorsements::PipelineConfiguration;
use verification_record_proto::{
    access_policy_proto::fcp::confidentialcompute::{
        AccessBudget, DataAccessPolicy, PipelineVariantPolicy, access_budget::Kind,
        data_access_policy, pipeline_variant_policy,
    },
    fcp::confidentialcompute::AttestationVerificationRecord,
    payload_transparency_proto::fcp::confidentialcompute::{SignedPayload, signed_payload},
};

/// Writes a human readable explanation for the given FCP
/// [`AttestationVerificationRecord`] into the given buffer.
pub async fn explain_record(
    buf: &mut dyn std::fmt::Write,
    record: &AttestationVerificationRecord,
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    writeln!(buf, "========================================")?;
    writeln!(buf, "======= KMS ATTESTATION EVIDENCE =======")?;
    writeln!(buf, "========================================")?;
    writeln!(buf)?;
    // Both `encryption_key` and `attestation_evidence` provide equivalent
    // information about the KMS, but only one will be set.
    if let Some(encryption_key) = &record.encryption_key {
        explain_encryption_key(buf, encryption_key, client)
            .await
            .context("failed to explain encryption key")?;
    } else if let Some(attestation_evidence) = &record.attestation_evidence {
        explain_attestation_evidence(buf, attestation_evidence)
            .context("failed to explain attestation evidence")?;
    } else {
        bail!("record is missing attestation evidence");
    }

    writeln!(buf)?;
    writeln!(buf, "========================================")?;
    writeln!(buf, "========== DATA ACCESS POLICY ==========")?;
    writeln!(buf, "========================================")?;
    writeln!(buf)?;
    // Both `pipeline_configuration` and `data_access_policy` provide equivalent
    // information about the pipeline, but only one will be set.
    if let Some(pipeline_configuration) = &record.pipeline_configuration {
        explain_pipeline_configuration(buf, pipeline_configuration, client)
            .await
            .context("failed to explain pipeline configuration")?;
    } else if let Some(data_access_policy) = &record.data_access_policy {
        explain_data_access_policy(buf, data_access_policy)
            .context("failed to explain data access policy")?;
    } else {
        bail!("record is missing data access policy");
    }
    Ok(())
}

/// Writes a human readable explanation for the given attestation evidence to
/// the given buffer.
fn explain_attestation_evidence(
    buf: &mut dyn std::fmt::Write,
    evidence: &verification_record_proto::evidence_proto::oak::attestation::v1::Evidence,
) -> anyhow::Result<()> {
    let evidence = Evidence::decode(evidence.encode_to_vec().as_slice())?;
    let extracted_evidence = oak_attestation_verification::extract_evidence(&evidence)
        .context("could not extract evidence data from provided Evidence proto")?;

    let write_link_to_oak = |buf: &mut dyn std::fmt::Write| -> anyhow::Result<()> {
        writeln!(
            buf,
            "Note: binaries for this layer are generally provided by the Oak project \
            (https://github.com/project-oak/oak)."
        )?;
        Ok(())
    };

    match extracted_evidence.evidence_values.context("extracted evidence missing EvidenceValues")? {
        EvidenceValues::OakRestrictedKernel(_) => {
            bail!(
                "Oak Restricted Kernel evidence is no longer supported. Please rerun this tool at \
                or before commit 4d130f31faa6dd0d0851b053bc818e5531165e76.",
            );
        }

        EvidenceValues::OakContainers(
            ref containers_data @ OakContainersData {
                ref root_layer,
                ref kernel_layer,
                ref system_layer,
                ref container_layer,
            },
        ) => {
            writeln!(buf, "{}", containers_data.title()?)?;
            writeln!(buf)?;
            let Some(root_layer) = root_layer else {
                bail!("missing root layer evidence");
            };
            let Some(kernel_layer) = kernel_layer else {
                bail!("missing kernel layer evidence");
            };
            let Some(system_layer) = system_layer else {
                bail!("missing system layer evidence");
            };
            let Some(container_layer) = container_layer else {
                bail!("missing container layer evidence");
            };
            writeln!(buf, "_____ {} _____", root_layer.title()?)?;
            writeln!(buf,)?;
            writeln!(buf, "{}", root_layer.description()?)?;
            writeln!(buf,)?;
            write_link_to_oak(buf)?;
            writeln!(buf,)?;
            writeln!(buf, "_____ {} _____", kernel_layer.title()?)?;
            writeln!(buf,)?;
            writeln!(buf, "{}", kernel_layer.description()?)?;
            writeln!(buf,)?;
            write_link_to_oak(buf)?;
            writeln!(buf,)?;
            writeln!(buf, "_____ {} _____", system_layer.title()?)?;
            writeln!(buf,)?;
            writeln!(buf, "{}", system_layer.description()?)?;
            writeln!(buf,)?;
            write_link_to_oak(buf)?;
            writeln!(buf,)?;
            writeln!(buf, "_____ {} _____", container_layer.title()?,)?;
            writeln!(buf,)?;
            writeln!(buf, "{}", container_layer.description()?)?;
            writeln!(buf,)?;
            writeln!(
                buf,
                "Note: this layer describes the \"KMS\" application binary, which is generally \
                a build of the `kms` in the \
                https://github.com/google-parfait/confidential-federated-compute repository.",
            )?;
            writeln!(buf,)?;
        }

        unexpected_evidence_type => {
            bail!(
                "Application evidence in an FCP attestation record is currently only expected to \
                describe Oak Containers applications (found the following evidence: \
                {unexpected_evidence_type:?})",
            );
        }
    }
    Ok(())
}

/// Writes a human readable explanation for the attestation evidence of the KMS
/// that generated the given encryption key to the given buffer.
async fn explain_encryption_key(
    buf: &mut dyn std::fmt::Write,
    encryption_key: &SignedPayload,
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    // Find the evidence digest from the SignedPayload signature headers.
    let mut endorsed_evidence_sha256 = None;
    let mut signed_payloads = vec![&encryption_key];
    while let Some(payload) = signed_payloads.pop() {
        for signature in &payload.signatures {
            let headers = signed_payload::signature::Headers::decode(&signature.headers[..])?;
            if let Some(oak_signature) = headers.oak_application_signature {
                let headers =
                    signed_payload::signature::Headers::decode(&oak_signature.headers[..])?;
                ensure!(
                    endorsed_evidence_sha256.is_none_or(|v| v == headers.endorsed_evidence_sha256),
                    "found conflicting endorsed evidence digests"
                );
                endorsed_evidence_sha256 = Some(headers.endorsed_evidence_sha256);
            }
        }
    }
    let endorsed_evidence_sha256 =
        endorsed_evidence_sha256.context("no endorsed evidence digest found")?;

    // Fetch the endorsed evidence.
    let url = format!(
        "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:{}",
        hex::encode(&endorsed_evidence_sha256)
    );
    writeln!(buf, "Downloading attestation evidence from {url}.")?;
    writeln!(buf)?;
    let response = client
        .get(url)
        .send()
        .and_then(|r| r.bytes())
        .await
        .context("failed to fetch attestation evidence from content-addressable storage")?;
    ensure!(
        &Sha256::digest(&response)[..] == &endorsed_evidence_sha256,
        "endorsed evidence digest does not match"
    );

    let endorsed_evidence = EndorsedEvidence::decode(&*response)?;
    explain_attestation_evidence(
        buf,
        &endorsed_evidence.evidence.context("endorsed evidence missing evidence")?,
    )
}

/// Writes a human readable explanation for the given data access policy to the
/// given buffer.
pub fn explain_data_access_policy(
    buf: &mut dyn std::fmt::Write,
    policy: &DataAccessPolicy,
) -> anyhow::Result<()> {
    if policy.pipelines.is_empty() {
        return explain_legacy_data_access_policy(buf, policy);
    }

    writeln!(buf, "The data access policy allows {} logical pipelines.", policy.pipelines.len(),)?;
    for (name, pipeline) in &policy.pipelines {
        writeln!(buf, "Logical pipeline '{name}' has {} instances.", pipeline.instances.len(),)?;
        for (i, instance) in pipeline.instances.iter().enumerate() {
            writeln!(buf, "Printing details of pipeline instance #{i}:")?;
            explain_pipeline_variant_policy(buf, instance)?;
        }
    }
    Ok(())
}

// TODO: This is almost exactly the same as the
// `explain_legacy_data_access_policy` function, except that the input message
// types don't match (even though they both have the same fields).
fn explain_pipeline_variant_policy(
    buf: &mut dyn std::fmt::Write,
    policy: &PipelineVariantPolicy,
) -> anyhow::Result<()> {
    writeln!(
        buf,
        "The data access policy allows {} data transformations and defines {} shared access \
        budgets.",
        policy.transforms.len(),
        policy.shared_access_budgets.len(),
    )?;

    for (i, transform) in policy.transforms.iter().enumerate() {
        writeln!(buf)?;
        writeln!(buf, ">>>>> Transform #{i} <<<<<",)?;
        writeln!(buf, "Source node IDs: {:?}", transform.src_node_ids)?;
        if !transform.dst_node_ids.is_empty() {
            writeln!(buf, "Destination node IDs: {:?}", transform.dst_node_ids)?;
        }
        writeln!(buf)?;
        if let Some(config_constraints) = &transform.config_constraints {
            writeln!(buf, "Configuration constraints: {}", config_constraints.type_url)?;
            // If we know the type of config constraint, then parse it and pretty-print it
            // for readability. Otherwise print the bytes as-is. They could be
            // parsed using another tool in that case.
            if let Ok(fed_sql_config_constraints) =
                config_constraints.to_msg::<FedSqlContainerConfigConstraints>()
            {
                writeln!(buf, "{:?}", fed_sql_config_constraints)?;
            } else {
                writeln!(buf, "{:?}", config_constraints)?;
            }
        } else {
            writeln!(buf, "Configuration constraints: None")?;
        }
        writeln!(buf)?;
        explain_transform_access_budgets(buf, transform, i, &policy.shared_access_budgets)?;
        writeln!(buf)?;
        let app_matcher = transform.application.clone().unwrap_or_default();
        writeln!(buf, "Application matcher for this transform:")?;
        writeln!(buf, "- Tag: {}", app_matcher.tag.unwrap_or_default())?;
        if let Some(config_properties) = app_matcher.config_properties {
            writeln!(buf, "- Binary configuration restrictions:")?;
            // Note: we simply print the StructMatcher's debug output for now. We may want
            // to format this information more clearly in the future.
            writeln!(buf, "  {:?}", config_properties)?;
        }
        if let Some(ref_vals) = app_matcher.reference_values {
            let ref_vals = ReferenceValues::decode(ref_vals.encode_to_vec().as_slice())?;
            writeln!(
                buf,
                "- Applications performing this transform must provide attestation evidence that \
                can be verified with the following reference values:"
            )?;
            writeln!(buf)?;
            writeln!(buf, "{}", ref_vals.title()?)?;
            writeln!(buf, "{}", ref_vals.description()?)?;
            writeln!(buf)?;
        } else {
            writeln!(
                buf,
                "- Any application can perform this transform (attestation evidence will not be \
                verified)."
            )?;
        }
    }

    Ok(())
}

/// Writes a human readable explanation for the given access budget to the given
/// buffer.
fn explain_transform_access_budgets(
    buf: &mut dyn std::fmt::Write,
    transform: &pipeline_variant_policy::Transform,
    transform_idx: usize,
    shared_budgets: &[AccessBudget],
) -> Result<(), anyhow::Error> {
    writeln!(
        buf,
        "Access budgets: the transform's access to its source blob is gated by *all* of the \
        following access rules:"
    )?;
    let mut access_restricted = false;
    if let Some(access_budget) = &transform.access_budget {
        access_restricted |= process_budget_access_time_with(access_budget, |times| {
            writeln!(
                buf,
                "- limited access budget (at most {times} times): the transform may only access its source \
                blob this many times."
            )?;
            Ok(())
        })?;
    }
    for shared_budget_id in &transform.shared_access_budget_indices {
        let shared_budget = shared_budgets.get(*shared_budget_id as usize).with_context(|| {
            format!(
                "transform {transform_idx} references non-existent shared access budget \
                        {shared_budget_id}"
            )
        })?;
        access_restricted |= process_budget_access_time_with(shared_budget, |shared_times| {
            writeln!(
                buf,
                "- limited shared access budget #{shared_budget_id} (at most {shared_times} \
                times): this and other transforms sharing this same budget may only access their \
                source blobs this many times combined."
            )?;
            Ok(())
        })?;
    }
    if !access_restricted {
        writeln!(
            buf,
            "- no access budget restrictions apply: this transform may access the source blob an \
            unlimited number of times."
        )?;
    }
    Ok(())
}

fn explain_legacy_data_access_policy(
    buf: &mut dyn std::fmt::Write,
    policy: &DataAccessPolicy,
) -> anyhow::Result<()> {
    // TODO: Update to print new-style policies correctly.
    writeln!(
        buf,
        "The data access policy allows {} data transformations and defines {} shared access \
        budgets.",
        policy.transforms.len(),
        policy.shared_access_budgets.len(),
    )?;

    for (i, transform) in policy.transforms.iter().enumerate() {
        writeln!(buf)?;
        writeln!(buf, ">>>>> Transform #{i} <<<<<",)?;
        writeln!(buf, "Source blob ID: {}", transform.src)?;
        writeln!(buf)?;
        explain_legacy_transform_access_budgets(buf, transform, i, &policy.shared_access_budgets)?;
        writeln!(buf)?;
        let app_matcher = transform.application.clone().unwrap_or_default();
        writeln!(buf, "Application matcher for this transform:")?;
        writeln!(buf, "- Tag: {}", app_matcher.tag.unwrap_or_default())?;
        if let Some(config_properties) = app_matcher.config_properties {
            writeln!(buf, "- Binary configuration restrictions:")?;
            // Note: we simply print the StructMatcher's debug output for now. We may want
            // to format this information more clearly in the future.
            writeln!(buf, "  {:?}", config_properties)?;
        }
        if let Some(ref_vals) = app_matcher.reference_values {
            let ref_vals = ReferenceValues::decode(ref_vals.encode_to_vec().as_slice())?;
            writeln!(
                buf,
                "- Applications performing this transform must provide attestation evidence that \
                can be verified with the following reference values:"
            )?;
            writeln!(buf)?;
            writeln!(buf, "{}", ref_vals.title()?)?;
            writeln!(buf, "{}", ref_vals.description()?)?;
            writeln!(buf)?;
            writeln!(
                buf,
                "Note: we don't print sigstore.dev links for the binary digests in this list of \
                reference values. You can construct such links manually using the following \
                template: https://search.sigstore.dev/?hash=${{SHA2_256_HASH}}. For the root layer \
                stage0 binary the reference value will list the SHA2-384 hash, in which case the \
                SHA2-256 hash of that SHA2-384 hash should be used for the sigstore.dev lookup."
            )?;
        } else {
            writeln!(
                buf,
                "- Any application can perform this transform (attestation evidence will not be \
                verified)."
            )?;
        }
    }

    Ok(())
}

/// Writes a human readable explanation for the given access budget to the given
/// buffer.
fn explain_legacy_transform_access_budgets(
    buf: &mut dyn std::fmt::Write,
    transform: &data_access_policy::Transform,
    transform_idx: usize,
    shared_budgets: &[AccessBudget],
) -> Result<(), anyhow::Error> {
    writeln!(
        buf,
        "Access budgets: the transform's access to its source blob is gated by *all* of the \
        following access rules:"
    )?;
    let mut access_restricted = false;
    if let Some(access_budget) = &transform.access_budget {
        access_restricted |= process_budget_access_time_with(access_budget, |times| {
            writeln!(
                buf,
                "- limited access budget (at most {times} times): the transform may only access its source \
                blob this many times."
            )?;
            Ok(())
        })?;
    }
    for shared_budget_id in &transform.shared_access_budget_indices {
        let shared_budget = shared_budgets.get(*shared_budget_id as usize).with_context(|| {
            format!(
                "transform {transform_idx} references non-existent shared access budget \
                        {shared_budget_id}"
            )
        })?;
        access_restricted |= process_budget_access_time_with(shared_budget, |shared_times| {
            writeln!(
                buf,
                "- limited shared access budget #{shared_budget_id} (at most {shared_times} \
                times): this and other transforms sharing this same budget may only access their \
                source blobs this many times combined."
            )?;
            Ok(())
        })?;
    }
    if !access_restricted {
        writeln!(
            buf,
            "- no access budget restrictions apply: this transform may access the source blob an \
            unlimited number of times."
        )?;
    }
    Ok(())
}

/// Writes a human readable explanation for the given PipelineConfiguration
/// message to the given budget.
async fn explain_pipeline_configuration(
    buf: &mut dyn std::fmt::Write,
    pipeline_configuration: &SignedPayload,
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    // The PipelineConfiguration contains a digest of the data access policy,
    // which can be used to download the full policy.
    let config = PipelineConfiguration::decode(pipeline_configuration.payload.as_slice())?;
    let url = format!(
        "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:{}",
        hex::encode(&config.access_policy_sha256)
    );
    writeln!(buf, "Downloading data access policy from {url}.")?;
    writeln!(buf)?;
    let response = client
        .get(url)
        .send()
        .and_then(|r| r.bytes())
        .await
        .context("failed to fetch data access policy from content-addressable storage")?;
    ensure!(
        &Sha256::digest(&response)[..] == &config.access_policy_sha256,
        "data access policy digest does not match"
    );

    let access_policy = DataAccessPolicy::decode(&response[..])?;
    explain_data_access_policy(buf, &access_policy)
}

/// Convenience function that calls the callback `f` with number of times the
/// [`AccessBudget`] allows a piece of data to be accessed by a given
/// application, if and only if the budget actually restricts the number of
/// accesses. The callback is not invoked if the budget does not restrict
/// the number of accesses.
///
/// Returns whether the number of accesses are restricted by the budget.
fn process_budget_access_time_with<F>(budget: &AccessBudget, mut f: F) -> anyhow::Result<bool>
where
    F: FnMut(u32) -> anyhow::Result<()>,
{
    match budget.kind {
        Some(Kind::Times(times)) => {
            f(times)?;
            Ok(true)
        }
        // Note: an access budget that allows unlimited access doesn't really restrict access, so we
        // don't need to report such access policies.
        None => Ok(false),
    }
}

/// Writes a human readable explanation for a SignedPayload signature structure
/// to the given buffer. The payload type of the signed payload is inferred from
/// context.
pub async fn explain_signed_payload(
    buf: &mut dyn std::fmt::Write,
    mut sig_structure: &[u8],
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    // Parse the SignedPayload signature structure.
    let len = sig_structure.read_varint()?;
    let (context, mut sig_structure) =
        sig_structure.split_at_checked(len).context("unexpected end of input1")?;
    ensure!(context == b"SignedPayload", "unexpected SignedPayload context string");
    let len = sig_structure.read_varint()?;
    let (headers, mut sig_structure) =
        sig_structure.split_at_checked(len).context("unexpected end of input2")?;
    let len = sig_structure.read_varint()?;
    let (payload, sig_structure) =
        sig_structure.split_at_checked(len).context("unexpected end of input3")?;
    ensure!(sig_structure.len() == 0, "unexpected trailing bytes");

    // Construct a SignedPayload message with the parsed fields and delegate
    // to the appropriate explain_* function. We assume that the payload is for
    // a KMS encryption key if the headers contain the Oak "Built from open
    // source" claim; this isn't a perfect heuristic, but it should be correct
    // for SignedPayload signature structures that are uploaded to transparency
    // logs.
    let signed_payload = SignedPayload {
        payload: payload.to_vec(),
        signatures: vec![signed_payload::Signature {
            headers: headers.to_vec(),
            ..Default::default()
        }],
    };
    if signed_payload::signature::Headers::decode(headers)?
        .claims
        .iter()
        .any(|c| c == "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md")
    {
        explain_encryption_key(buf, &signed_payload, client).await
    } else {
        explain_pipeline_configuration(buf, &signed_payload, client).await
    }
}
