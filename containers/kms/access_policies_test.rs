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

use access_policies::validate_pipeline_invocation_policies;
use access_policy_proto::fcp::confidentialcompute::{
    pipeline_variant_policy::Transform, ApplicationMatcher,
    DataAccessPolicy as AuthorizedLogicalPipelinePolicies, LogicalPipelinePolicy,
    PipelineVariantPolicy,
};
use googletest::prelude::*;
use prost::Message;

/// Builds a PipelineVariantPolicy with the given source transform. This is not
/// a representative policy, but it's sufficient for testing.
fn build_test_variant_policy(src: u32) -> PipelineVariantPolicy {
    PipelineVariantPolicy {
        transforms: vec![Transform { src, ..Default::default() }],
        ..Default::default()
    }
}

#[googletest::test]
fn validate_pipeline_invocation_policies_success() {
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy {
                instances: vec![
                    build_test_variant_policy(1),
                    build_test_variant_policy(2),
                    build_test_variant_policy(3),
                ],
            },
        )]
        .into(),
        ..Default::default()
    };

    expect_that!(
        validate_pipeline_invocation_policies(
            "test",
            &build_test_variant_policy(2).encode_to_vec(),
            &[logical_pipeline_policies.encode_to_vec()],
        ),
        ok(anything()),
    );
}

#[googletest::test]
fn validate_pipeline_invocation_policies_fails_with_unsupported_fields() {
    const LOGICAL_PIPELINE_NAME: &str = "test";

    fn build_authorized_logical_pipeline_policies(
        variant: PipelineVariantPolicy,
    ) -> AuthorizedLogicalPipelinePolicies {
        AuthorizedLogicalPipelinePolicies {
            pipelines: [(
                LOGICAL_PIPELINE_NAME.into(),
                LogicalPipelinePolicy { instances: vec![variant] },
            )]
            .into(),
            ..Default::default()
        }
    }

    // ApplicationMatcher.config_properties is not supported.
    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            application: Some(ApplicationMatcher {
                config_properties: Some(Default::default()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("config_properties is not supported")))
    );

    // Transform.access_budget is not supported.
    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            access_budget: Some(Default::default()),
            ..Default::default()
        }],
        ..Default::default()
    };
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("access_budget is not supported")))
    );

    // Transform.shared_access_budget_indices is not supported.
    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform { shared_access_budget_indices: vec![0], ..Default::default() }],
        ..Default::default()
    };
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("shared_access_budget_indices are not supported")))
    );

    // dst_node_id 0 is not allowed.
    let variant_policy = PipelineVariantPolicy {
        transforms: vec![Transform { dst_node_ids: vec![0, 1, 2], ..Default::default() }],
        ..Default::default()
    };
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("0 is not a valid dst_node_id")))
    );
}

#[googletest::test]
fn validate_pipeline_invocation_policies_fails_without_logical_pipeline_policy() {
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![build_test_variant_policy(1)] },
        )]
        .into(),
        ..Default::default()
    };

    expect_that!(
        validate_pipeline_invocation_policies(
            "other",
            &build_test_variant_policy(1).encode_to_vec(),
            &[logical_pipeline_policies.encode_to_vec()],
        ),
        err(displays_as(contains_substring(
            "PipelineVariantPolicy not found in AuthorizedLogicalPipelinePolicies 0",
        ))),
    );
}

#[googletest::test]
fn validate_pipeline_invocation_policies_fails_without_variant_policy() {
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [
            (
                "test".into(),
                LogicalPipelinePolicy { instances: vec![build_test_variant_policy(1)] },
            ),
            (
                "other".into(),
                LogicalPipelinePolicy { instances: vec![build_test_variant_policy(2)] },
            ),
        ]
        .into(),
        ..Default::default()
    };

    expect_that!(
        validate_pipeline_invocation_policies(
            "test",
            &build_test_variant_policy(2).encode_to_vec(),
            &[logical_pipeline_policies.encode_to_vec()],
        ),
        err(displays_as(contains_substring(
            "PipelineVariantPolicy not found in AuthorizedLogicalPipelinePolicies 0"
        )))
    );
}

#[googletest::test]
fn validate_pipeline_invocation_policies_fails_without_all_policies_matching() {
    let logical_pipeline_policies1 = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![build_test_variant_policy(1)] },
        )]
        .into(),
        ..Default::default()
    };
    let logical_pipeline_policies2 = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![build_test_variant_policy(2)] },
        )]
        .into(),
        ..Default::default()
    };

    expect_that!(
        validate_pipeline_invocation_policies(
            "test",
            &build_test_variant_policy(1).encode_to_vec(),
            &[
                logical_pipeline_policies1.encode_to_vec(),
                logical_pipeline_policies2.encode_to_vec(),
            ],
        ),
        err(displays_as(contains_substring(
            "PipelineVariantPolicy not found in AuthorizedLogicalPipelinePolicies 1"
        )))
    );
}

#[googletest::test]
fn validate_pipeline_invocation_policies_fails_with_malformed_policies() {
    let variant_policy = build_test_variant_policy(1).encode_to_vec();
    let logical_pipeline_policies = AuthorizedLogicalPipelinePolicies {
        pipelines: [(
            "test".into(),
            LogicalPipelinePolicy { instances: vec![build_test_variant_policy(1)] },
        )]
        .into(),
        ..Default::default()
    }
    .encode_to_vec();

    let buffer = [variant_policy.as_slice(), b"...."].concat();
    for i in (1..buffer.len()).filter(|i| *i != variant_policy.len()) {
        let partial_variant_policy = &buffer[..i];
        expect_that!(
            validate_pipeline_invocation_policies(
                "test",
                partial_variant_policy,
                &[logical_pipeline_policies.as_slice()],
            ),
            err(displays_as(contains_substring("failed to decode PipelineVariantPolicy"))),
            "i = {}/{}",
            i,
            variant_policy.len()
        );
    }

    let buffer = [logical_pipeline_policies.as_slice(), b"...."].concat();
    for i in (0..buffer.len()).filter(|i| *i != logical_pipeline_policies.len()) {
        let partial_logical_pipeline_policies = &buffer[..i];
        expect_that!(
            validate_pipeline_invocation_policies(
                "test",
                &variant_policy,
                &[partial_logical_pipeline_policies],
            ),
            err(displays_as(contains_substring("AuthorizedLogicalPipelinePolicies 0"))),
            "i = {}/{}",
            i,
            logical_pipeline_policies.len()
        );
    }
}
