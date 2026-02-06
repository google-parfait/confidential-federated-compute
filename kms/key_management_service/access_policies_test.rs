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

use access_policies::{
    authorize_transform, validate_pipeline_invocation_policies, AuthorizedTransform,
};
use access_policy_proto::{
    any_proto::google::protobuf::Any,
    fcp::confidentialcompute::{
        pipeline_variant_policy::Transform, ApplicationMatcher,
        DataAccessPolicy as AuthorizedLogicalPipelinePolicies, LogicalPipelinePolicy,
        PipelineVariantPolicy,
    },
    reference_value_proto::oak::attestation::v1::ReferenceValues,
};
use googletest::prelude::*;
use prost::Message;
use session_test_utils::{get_test_endorsements, get_test_evidence, get_test_reference_values};

/// Builds a PipelineVariantPolicy with the given source transform. This is not
/// a representative policy, but it's sufficient for testing.
fn build_test_variant_policy(src: u32) -> PipelineVariantPolicy {
    PipelineVariantPolicy {
        transforms: vec![Transform {
            src_node_ids: vec![src],
            application: Some(ApplicationMatcher {
                reference_values: Some(get_test_reference_values()),
                ..Default::default()
            }),
            ..Default::default()
        }],
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

    // ApplicationMatcher.reference_values is required.
    let mut variant_policy = build_test_variant_policy(1);
    variant_policy.transforms[0].application.as_mut().unwrap().reference_values = None;
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("reference_values are required")))
    );

    // ApplicationMatcher.config_properties is not supported.
    let mut variant_policy = build_test_variant_policy(1);
    variant_policy.transforms[0].application.as_mut().unwrap().config_properties =
        Some(Default::default());
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("config_properties is not supported")))
    );

    // Transform.access_budget is not supported.
    let mut variant_policy = build_test_variant_policy(1);
    variant_policy.transforms[0].access_budget = Some(Default::default());
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("access_budget is not supported")))
    );

    // Transform.shared_access_budget_indices is not supported.
    let mut variant_policy = build_test_variant_policy(1);
    variant_policy.transforms[0].shared_access_budget_indices = vec![0];
    expect_that!(
        validate_pipeline_invocation_policies(
            LOGICAL_PIPELINE_NAME,
            &variant_policy.encode_to_vec(),
            &[build_authorized_logical_pipeline_policies(variant_policy).encode_to_vec()],
        ),
        err(displays_as(contains_substring("shared_access_budget_indices are not supported")))
    );

    // dst_node_id 0 is not allowed.
    let mut variant_policy = build_test_variant_policy(1);
    variant_policy.transforms[0].dst_node_ids = vec![0, 1, 2];
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

#[googletest::test]
fn authorize_transform_success() {
    let policy = PipelineVariantPolicy {
        transforms: vec![
            Transform {
                src_node_ids: vec![1, 2],
                dst_node_ids: vec![3, 4],
                application: Some(ApplicationMatcher {
                    tag: Some("tagA".into()),
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                config_constraints: Some(Any { value: b"config1".into(), ..Default::default() }),
                ..Default::default()
            },
            Transform {
                src_node_ids: vec![5, 6],
                dst_node_ids: vec![7, 8],
                application: Some(ApplicationMatcher {
                    tag: Some("tagB".into()),
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                config_constraints: Some(Any { value: b"config2".into(), ..Default::default() }),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    expect_that!(
        authorize_transform(
            &policy.encode_to_vec(),
            &get_test_evidence(),
            &get_test_endorsements(),
            "tagB",
            &Default::default(),
        ),
        ok(matches_pattern!(AuthorizedTransform {
            index: eq(1),
            src_node_ids: elements_are!(eq(5), eq(6)),
            dst_node_ids: elements_are!(eq(7), eq(8)),
            config_constraints: some(matches_pattern!(Any { value: eq(b"config2") })),
            encryption_public_key: not(empty()),
            signing_public_key: not(empty()),
        }))
    );
}

#[googletest::test]
fn authorize_transform_without_explicit_reference_values() {
    let policy = PipelineVariantPolicy {
        transforms: vec![Transform {
            application: Some(ApplicationMatcher { reference_values: None, ..Default::default() }),
            ..Default::default()
        }],
        ..Default::default()
    };

    expect_that!(
        authorize_transform(
            &policy.encode_to_vec(),
            &get_test_evidence(),
            &get_test_endorsements(),
            "tag",
            &Default::default(),
        ),
        err(displays_as(contains_substring("no transforms matched")))
    );
}

#[googletest::test]
fn authorize_transform_returns_first_match() {
    let policy = PipelineVariantPolicy {
        transforms: vec![
            Transform {
                src_node_ids: vec![1],
                application: Some(ApplicationMatcher {
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Transform {
                src_node_ids: vec![2],
                application: Some(ApplicationMatcher {
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Transform {
                src_node_ids: vec![3],
                application: Some(ApplicationMatcher {
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    expect_that!(
        authorize_transform(
            &policy.encode_to_vec(),
            &get_test_evidence(),
            &get_test_endorsements(),
            "tag",
            &Default::default(),
        ),
        ok(matches_pattern!(AuthorizedTransform {
            index: eq(0),
            src_node_ids: elements_are!(eq(1)),
        }))
    );
}

#[googletest::test]
fn authorize_transform_fails_with_invalid_policy() {
    expect_that!(
        authorize_transform(
            b"invalid-policy",
            &get_test_evidence(),
            &get_test_endorsements(),
            "tag",
            &Default::default(),
        ),
        err(displays_as(contains_substring("failed to decode PipelineVariantPolicy")))
    );
}

#[googletest::test]
fn authorize_transform_fails_without_match() {
    let policy = PipelineVariantPolicy {
        transforms: vec![
            Transform {
                src_node_ids: vec![1, 2],
                dst_node_ids: vec![3, 4],
                application: Some(ApplicationMatcher {
                    tag: Some("tagA".into()),
                    reference_values: Some(get_test_reference_values()),
                    ..Default::default()
                }),
                config_constraints: Some(Any { value: b"config1".into(), ..Default::default() }),
                ..Default::default()
            },
            Transform {
                src_node_ids: vec![5, 6],
                dst_node_ids: vec![7, 8],
                application: Some(ApplicationMatcher {
                    tag: Some("tagB".into()),
                    reference_values: Some(ReferenceValues::default()), // Empty RV are invalid.
                    ..Default::default()
                }),
                config_constraints: Some(Any { value: b"config2".into(), ..Default::default() }),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    expect_that!(
        authorize_transform(
            &policy.encode_to_vec(),
            &get_test_evidence(),
            &get_test_endorsements(),
            "tagB",
            &Default::default(),
        ),
        err(displays_as(matches_regex(
            "(?ms).*no transforms matched.+tag mismatch.+unsupported ReferenceValues.*"
        )))
    );
}
