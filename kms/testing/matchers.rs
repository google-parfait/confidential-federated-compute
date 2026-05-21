// Copyright 2025 Google LLC.
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

use std::{fmt::Debug, marker::PhantomData};

use googletest::{
    description::Description,
    matcher::{Matcher, MatcherBase, MatcherResult},
};
use prost::Message;

/// Matches an error that has the given code in its context.
pub fn code(expected: tonic::Code) -> impl for<'a> Matcher<&'a anyhow::Error> {
    CodeMatcher { expected }
}

#[derive(MatcherBase)]
struct CodeMatcher {
    expected: tonic::Code,
}

impl<'a> Matcher<&'a anyhow::Error> for CodeMatcher {
    fn matches(&self, actual: &'a anyhow::Error) -> MatcherResult {
        actual.downcast_ref::<tonic::Code>().map(|c| *c == self.expected).unwrap_or(false).into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => format!("has code {}", self.expected).into(),
            MatcherResult::NoMatch => format!("doesn't have code {}", self.expected).into(),
        }
    }
}

/// Matches an error that has a context or root cause matched by `inner`.
pub fn has_context<InnerMatcher: for<'a> Matcher<&'a str>>(
    inner: InnerMatcher,
) -> impl for<'a> Matcher<&'a anyhow::Error> {
    ContextMatcher { inner }
}

#[derive(MatcherBase)]
pub struct ContextMatcher<InnerMatcher> {
    inner: InnerMatcher,
}

impl<'a, InnerMatcher: for<'b> Matcher<&'b str>> Matcher<&'a anyhow::Error>
    for ContextMatcher<InnerMatcher>
{
    fn matches(&self, actual: &'a anyhow::Error) -> MatcherResult {
        actual.chain().any(|e| self.inner.matches(&format!("{e}")).into()).into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => Description::new()
                .text("has a context which")
                .nested(self.inner.describe(MatcherResult::Match)),
            MatcherResult::NoMatch => Description::new()
                .text("doesn't have a context which")
                .nested(self.inner.describe(MatcherResult::Match)),
        }
    }
}

/// Matches a `Vec<u8>` that deserializes to a message matched by `inner`.
pub fn when_deserialized<T: Message + Debug + Default, InnerMatcher: for<'a> Matcher<&'a T>>(
    inner: InnerMatcher,
) -> SerializedMessageMatcher<InnerMatcher, T> {
    SerializedMessageMatcher { inner, phantom: PhantomData }
}

#[derive(MatcherBase)]
pub struct SerializedMessageMatcher<InnerMatcher, MessageT> {
    inner: InnerMatcher,
    phantom: PhantomData<MessageT>,
}

impl<'a, ActualT, MessageT, InnerMatcher> Matcher<&'a ActualT>
    for SerializedMessageMatcher<InnerMatcher, MessageT>
where
    ActualT: AsRef<[u8]> + Debug,
    MessageT: Message + Debug + Default,
    InnerMatcher: for<'b> Matcher<&'b MessageT>,
{
    fn matches(&self, actual: &'a ActualT) -> MatcherResult {
        MessageT::decode(actual.as_ref()).map(|m| self.inner.matches(&m)).unwrap_or(false.into())
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => Description::new()
                .text("deserializes to a message which")
                .nested(self.inner.describe(MatcherResult::Match)),
            MatcherResult::NoMatch => Description::new()
                .text("doesn't deserialize to a message which")
                .nested(self.inner.describe(MatcherResult::Match)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::anyhow;
    use googletest::prelude::*;
    use prost_types::Duration;
    use tonic::Code::{Internal, Unknown};

    #[gtest]
    fn code_matches_top_level_context() {
        expect_that!(anyhow!("message").context(Internal), code(Internal));
    }

    #[gtest]
    fn code_matches_nested_context() {
        expect_that!(
            anyhow!("message").context("context").context(Internal).context("extra context"),
            code(Internal)
        );
    }

    #[gtest]
    fn code_matches_updated_code() {
        expect_that!(anyhow!("message").context(Internal).context(Unknown), code(Unknown));
    }

    #[gtest]
    fn code_does_not_match_different_code() {
        expect_that!(anyhow!("message").context(Unknown), not(code(Internal)));
    }

    #[gtest]
    fn code_does_not_match_overwritten_code() {
        expect_that!(anyhow!("message").context(Internal).context(Unknown), not(code(Internal)));
    }

    #[gtest]
    fn code_does_not_match_no_code() {
        expect_that!(anyhow!("message"), not(code(Internal)));
    }

    #[gtest]
    fn has_context_matches_top_level_context() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("context2"))
        );
    }

    #[gtest]
    fn has_context_matches_intermediate_context() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("context1"))
        );
    }

    #[gtest]
    fn has_context_matches_root_cause() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("message"))
        );
    }

    #[gtest]
    fn has_context_does_not_match_different_string() {
        expect_that!(anyhow!("message").context("context1"), not(has_context(eq("context2"))));
    }

    #[gtest]
    fn when_deserialized_matches() {
        expect_that!(
            Duration { seconds: 1, ..Default::default() }.encode_to_vec(),
            when_deserialized(matches_pattern!(Duration { seconds: &1, .. }))
        );
    }

    #[gtest]
    fn when_deserialized_does_not_match_invalid_message() {
        expect_that!(b"invalid", not(when_deserialized::<Duration, _>(anything())));
    }

    #[gtest]
    fn when_deserialized_does_not_match_different_message() {
        expect_that!(
            Duration { seconds: 1, ..Default::default() }.encode_to_vec(),
            not(when_deserialized(matches_pattern!(Duration { seconds: &2, .. })))
        );
    }
}
