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
    DataAccessPolicyWithSerializedVariants as AuthorizedLogicalPipelinePoliciesWithSerializedVariants,
    PipelineVariantPolicy,
};
use anyhow::{anyhow, ensure, Context};
use prost::Message;

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
    policy.transforms.into_iter().try_for_each(|transform| {
        ensure!(
            transform.application.and_then(|app| app.config_properties).is_none(),
            "config_properties is not supported"
        );
        ensure!(transform.access_budget.is_none(), "access_budget is not supported");
        ensure!(
            transform.shared_access_budget_indices.is_empty(),
            "shared_access_budget_indices are not supported"
        );
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
