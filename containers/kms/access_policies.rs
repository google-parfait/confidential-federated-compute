// Copyright 2025 The Trusted Computations Platform Authors.
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

use access_policy_proto::fcp::confidentialcompute::{
    pipeline_variant_policy::Transform,
    DataAccessPolicyWithSerializedVariants as AuthorizedLogicalPipelinePoliciesWithSerializedVariants,
    PipelineVariantPolicy,
};
use anyhow::{anyhow, ensure, Context};
use kms_proto::{
    any_proto::google::protobuf::Any,
    endorsement_proto::oak::attestation::v1::Endorsements as KmsEndorsements,
    evidence_proto::oak::attestation::v1::Evidence as KmsEvidence,
    timestamp_proto::google::protobuf::Timestamp,
};
use oak_attestation_verification::verifier::verify;
use oak_proto_rust::oak::attestation::v1::{Endorsements, Evidence, ExtractedEvidence};
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;

/// Validates the policies provided to AuthorizeLogicalPipeline.
///
/// * The PipelineVariantPolicy must not contain any unsupported fields.
/// * The PipelineVariantPolicy must appear in each of the
///   AuthorizedLogicalPipelinePolicies messages.
pub fn validate_pipeline_invocation_policies(
    logical_pipeline_name: &str,
    pipeline_variant_policy: &[u8],
    authorized_logical_pipeline_policies: &[impl std::ops::Deref<Target = [u8]>],
) -> anyhow::Result<()> {
    // Verify that the PipelineVariantPolicy can be decoded and doesn't contain
    // any unsupported fields.
    let policy = PipelineVariantPolicy::decode(pipeline_variant_policy)
        .context("failed to decode PipelineVariantPolicy")?;
    policy.transforms.iter().try_for_each(|transform| {
        ensure!(
            transform.application.as_ref().and_then(|app| app.reference_values.as_ref()).is_some(),
            "reference_values are required"
        );
        ensure!(
            transform.application.as_ref().and_then(|app| app.config_properties.as_ref()).is_none(),
            "config_properties is not supported"
        );
        ensure!(transform.access_budget.is_none(), "access_budget is not supported");
        ensure!(
            transform.shared_access_budget_indices.is_empty(),
            "shared_access_budget_indices are not supported"
        );
        ensure!(!transform.dst_node_ids.contains(&0), "0 is not a valid dst_node_id");
        Ok(())
    })?;

    // Verify that the PipelineVariantPolicy appears in each of the
    // AuthorizedLogicalPipelinePolicies messages.
    for (i, entry) in authorized_logical_pipeline_policies.iter().enumerate() {
        let policies = AuthorizedLogicalPipelinePoliciesWithSerializedVariants::decode(&entry[..])
            .with_context(|| format!("failed to decode AuthorizedLogicalPipelinePolicies {}", i))?;
        policies
            .pipelines
            .get(logical_pipeline_name)
            .and_then(|logical_pipeline_policy| {
                logical_pipeline_policy
                    .instances
                    .iter()
                    .find(|policy| *policy == pipeline_variant_policy)
            })
            .ok_or_else(|| {
                anyhow!(
                    "PipelineVariantPolicy not found in AuthorizedLogicalPipelinePolicies {}",
                    i
                )
            })?;
    }

    Ok(())
}

/// Information about the Transform in the access policy that authorized the
/// requestor.
#[derive(Debug)]
pub struct AuthorizedTransform {
    /// The index of the matched transform in the PipelineVariantPolicy.
    pub index: usize,
    /// The node ID of the data the transform is authorized to access.
    pub src_node_ids: Vec<u32>,
    /// The node ID of the data the transform is authorized to produce.
    pub dst_node_ids: Vec<u32>,
    /// Any configuration constraints that should be applied by the transform.
    pub config_constraints: Option<Any>,
    /// The evidence values extracted during attestation verification.
    pub extracted_evidence: ExtractedEvidence,
}

/// Attempts to match the requestor against the Transforms in the access policy.
pub fn authorize_transform(
    pipeline_variant_policy: &[u8],
    evidence: &KmsEvidence,
    endorsements: &KmsEndorsements,
    tag: &str,
    now: &Timestamp,
) -> anyhow::Result<AuthorizedTransform> {
    let policy = PipelineVariantPolicy::decode(pipeline_variant_policy)
        .context("failed to decode PipelineVariantPolicy")?;
    let evidence = evidence.convert()?;
    let endorsements = endorsements.convert()?;
    let now_utc_millis =
        now.seconds.saturating_mul(1000).saturating_add((now.nanos as i64) / 1_000_000);

    // Check each transform, stopping as soon as a match is found. For
    // non-matching transforms, collect the failure reason.
    //
    // We abuse Result to implement this short-circuiting, where the Ok and Err
    // types are swapped. This could be more cleanly implemented using
    // std::ops::ControlFlow once it's stable.
    let auth_results: Result<Vec<anyhow::Error>, AuthorizedTransform> = policy
        .transforms
        .into_iter()
        .enumerate()
        .map(|(index, transform)| {
            match match_transform(index, transform, &evidence, &endorsements, tag, now_utc_millis) {
                Ok(authorized_transform) => Err(authorized_transform),
                Err(err) => Ok(err),
            }
        })
        .collect();
    match auth_results {
        Err(authorized_transform) => Ok(authorized_transform),
        Ok(match_errors) => Err(anyhow!("no transforms matched: {:#?}", match_errors)),
    }
}

/// Attempts to match the requestor against a single Transform.
fn match_transform(
    index: usize,
    transform: Transform,
    evidence: &Evidence,
    endorsements: &Endorsements,
    tag: &str,
    now_utc_millis: i64,
) -> anyhow::Result<AuthorizedTransform> {
    let app = transform.application.unwrap_or_default();
    if let Some(app_tag) = app.tag {
        ensure!(app_tag == tag, "tag mismatch");
    }

    // Since we need ExtractedEvidence, we need to verify the attestation even
    // if the ApplicationMatcher doesn't contain any ReferenceValues. This
    // effectively makes ReferenceValues required for all transforms.
    let reference_values = app.reference_values.unwrap_or_default().convert()?;
    let extracted_evidence = verify(now_utc_millis, evidence, endorsements, &reference_values)
        .context("reference_values mismatch")?;

    // During migration, fall back to `src` if `src_node_ids` is empty.
    let mut src_node_ids = transform.src_node_ids;
    if src_node_ids.is_empty() {
        src_node_ids.push(transform.src);
    }

    Ok(AuthorizedTransform {
        index,
        src_node_ids,
        dst_node_ids: transform.dst_node_ids,
        config_constraints: transform.config_constraints,
        extracted_evidence,
    })
}
