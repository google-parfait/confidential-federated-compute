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

use std::sync::Arc;

use access_policy_proto::fcp::confidentialcompute::{
    pipeline_variant_policy::Transform,
    DataAccessPolicyWithSerializedVariants as AuthorizedLogicalPipelinePoliciesWithSerializedVariants,
    PipelineVariantPolicy,
};
use anyhow::{anyhow, bail, ensure, Context};
use kms_proto::{
    any_proto::google::protobuf::Any,
    endorsement_proto::oak::attestation::v1::Endorsements as KmsEndorsements,
    evidence_proto::oak::attestation::v1::Evidence as KmsEvidence,
    timestamp_proto::google::protobuf::Timestamp,
};
use oak_attestation_verification::{
    results::{unique_hybrid_encryption_public_key, unique_signing_public_key},
    AmdSevSnpDiceAttestationVerifier, AmdSevSnpPolicy, ContainerPolicy, FirmwarePolicy,
    InsecureAttestationVerifier, KernelPolicy, SystemPolicy,
};
use oak_attestation_verification_types::verifier::AttestationVerifier;
use oak_proto_rust::oak::attestation::v1::{
    attestation_results, reference_values, AmdSevReferenceValues, Endorsements, Evidence,
    OakContainersReferenceValues, ReferenceValues, RootLayerReferenceValues,
};
use oak_time::{clock::FixedClock, Instant};
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
    /// The encryption public key extracted during attestation verification.
    pub encryption_public_key: Vec<u8>,
    /// The signing public key extracted during attestation verification.
    pub signing_public_key: Vec<u8>,
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

    // Since we need signing and encryption keys, we need to verify the
    // attestation even if the ApplicationMatcher doesn't contain any
    // ReferenceValues. This effectively makes ReferenceValues required for all
    // transforms.
    let verifier =
        get_verifier(&app.reference_values.unwrap_or_default().convert()?, now_utc_millis)?;
    let results = verifier.verify(evidence, endorsements).context("reference_values mismatch")?;
    ensure!(
        results.status == attestation_results::Status::Success as i32,
        "attestation verification failed: {:?}: {}",
        results.status,
        results.reason
    );

    let encryption_public_key = unique_hybrid_encryption_public_key(&results)
        .map_err(|msg| anyhow!("evidence missing unique encryption public key: {}", msg))?;
    let signing_public_key = unique_signing_public_key(&results)
        .map_err(|msg| anyhow!("evidence missing unique signing public key: {}", msg))?;
    Ok(AuthorizedTransform {
        index,
        src_node_ids: transform.src_node_ids,
        dst_node_ids: transform.dst_node_ids,
        config_constraints: transform.config_constraints,
        encryption_public_key: encryption_public_key.clone(),
        signing_public_key: signing_public_key.clone(),
    })
}

/// Returns an AttestationVerifier for the given ReferenceValues.
fn get_verifier(
    reference_values: &ReferenceValues,
    now_utc_millis: i64,
) -> anyhow::Result<Box<dyn AttestationVerifier>> {
    match &reference_values.r#type {
        // Oak Containers (insecure)
        Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues { insecure: Some(_), .. }),
            kernel_layer: Some(kernel_ref_vals),
            system_layer: Some(system_ref_vals),
            container_layer: Some(container_ref_vals),
        })) => Ok(Box::new(InsecureAttestationVerifier::new(
            Arc::new(FixedClock::at_instant(Instant::from_unix_millis(now_utc_millis))),
            vec![
                Box::new(KernelPolicy::new(kernel_ref_vals)),
                Box::new(SystemPolicy::new(system_ref_vals)),
                Box::new(ContainerPolicy::new(container_ref_vals)),
            ],
        ))),

        // Oak Containers (AMD SEV-SNP)
        Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer:
                Some(RootLayerReferenceValues {
                    amd_sev:
                        Some(
                            amd_sev_ref_vals @ AmdSevReferenceValues {
                                stage0: Some(stage0_ref_vals),
                                ..
                            },
                        ),
                    insecure: None,
                    ..
                }),
            kernel_layer: Some(kernel_ref_vals),
            system_layer: Some(system_ref_vals),
            container_layer: Some(container_ref_vals),
        })) => Ok(Box::new(AmdSevSnpDiceAttestationVerifier::new(
            AmdSevSnpPolicy::new(amd_sev_ref_vals),
            Box::new(FirmwarePolicy::new(stage0_ref_vals)),
            vec![
                Box::new(KernelPolicy::new(kernel_ref_vals)),
                Box::new(SystemPolicy::new(system_ref_vals)),
                Box::new(ContainerPolicy::new(container_ref_vals)),
            ],
            Arc::new(FixedClock::at_instant(Instant::from_unix_millis(now_utc_millis))),
        ))),

        _ => bail!("unsupported ReferenceValues"),
    }
}
