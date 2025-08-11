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

use anyhow::{bail, Context};
use federated_compute::proto::access_budget::Kind;
use federated_compute::proto::{
    AccessBudget, AttestationVerificationRecord, DataAccessPolicy, PipelineVariantPolicy,
};
use oak_attestation_explain::{HumanReadableExplanation, HumanReadableTitle};
use oak_proto_rust::oak::attestation::v1::{
    extracted_evidence::EvidenceValues, Evidence, OakRestrictedKernelData,
};

/// Writes a human readable explanation for the given FCP
/// [`AttestationVerificationRecord`] into the given buffer.
pub fn explain_record(
    buf: &mut dyn std::fmt::Write,
    record: &AttestationVerificationRecord,
) -> anyhow::Result<()> {
    writeln!(buf, "========================================")?;
    writeln!(buf, "===== LEDGER ATTESTATION EVIDENCE ======")?;
    writeln!(buf, "========================================")?;
    writeln!(buf)?;
    explain_ledger_evidence(
        buf,
        record.attestation_evidence.as_ref().context("record is missing attestation evidence")?,
    )
    .context("failed to explain ledger attestation evidence")?;

    writeln!(buf)?;
    writeln!(buf, "========================================")?;
    writeln!(buf, "========== DATA ACCESS POLICY ==========")?;
    writeln!(buf, "========================================")?;
    writeln!(buf)?;

    explain_data_access_policy(
        buf,
        record.data_access_policy.as_ref().context("record is missing data access policy")?,
    )
    .context("failed to explain data access policy")?;
    Ok(())
}

/// Writes a human readable explanation for the given ledger attestation
/// evidence to the given buffer.
fn explain_ledger_evidence(
    buf: &mut dyn std::fmt::Write,
    evidence: &Evidence,
) -> anyhow::Result<()> {
    let extracted_ledger_evidence = oak_attestation_verification::extract_evidence(evidence)
        .context("could not extract evidence data from provided Evidence proto")?;

    let write_link_to_oak = |buf: &mut dyn std::fmt::Write| -> anyhow::Result<()> {
        writeln!(
            buf,
            "Note: binaries for this layer are generally provided by the Oak project \
            (https://github.com/project-oak/oak)."
        )?;
        Ok(())
    };

    match extracted_ledger_evidence
        .evidence_values
        .context("extracted evidence missing EvidenceValues")?
    {
        EvidenceValues::OakRestrictedKernel(
            ref restricted_kernel_data @ OakRestrictedKernelData {
                ref root_layer,
                ref kernel_layer,
                ref application_layer,
            },
        ) => {
            writeln!(buf, "{}", restricted_kernel_data.title()?)?;
            writeln!(buf)?;
            let Some(root_layer) = root_layer else {
                bail!("missing root layer evidence");
            };
            let Some(kernel_layer) = kernel_layer else {
                bail!("missing kernel layer evidence");
            };
            let Some(application_layer) = application_layer else {
                bail!("missing application layer evidence");
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
            writeln!(buf, "_____ {} _____", application_layer.title()?,)?;
            writeln!(buf,)?;
            writeln!(buf, "{}", application_layer.description()?)?;
            writeln!(buf,)?;
            writeln!(
                buf,
                "Note: this layer describes the \"ledger\" application binary, which is generally \
                a build of the `ledger_enclave_app` in the \
                https://github.com/google-parfait/confidential-federated-compute repository.",
            )?;
            writeln!(buf,)?;
        }
        unexpected_evidence_type => {
            bail!(
                "Ledger evidence in an FCP attestation record is currently only expected to \
                describe Oak Restricted Kernel applications (found the following evidence:
                {unexpected_evidence_type:?})",
            );
        }
    }
    Ok(())
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
        writeln!(buf, "Source blob ID: {}", transform.src)?;
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
    transform: &federated_compute::proto::pipeline_variant_policy::Transform,
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
    transform: &federated_compute::proto::data_access_policy::Transform,
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
