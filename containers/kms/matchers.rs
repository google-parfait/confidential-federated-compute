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

use googletest::{
    description::Description,
    matcher::{Matcher, MatcherResult},
};

/// Matches an error that has the given code in its context.
pub fn code(expected: tonic::Code) -> impl Matcher<ActualT = anyhow::Error> {
    CodeMatcher { expected }
}

struct CodeMatcher {
    expected: tonic::Code,
}

impl Matcher for CodeMatcher {
    type ActualT = anyhow::Error;

    fn matches(&self, actual: &Self::ActualT) -> MatcherResult {
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
pub fn has_context<InnerMatcherT: Matcher<ActualT = String>>(
    inner: InnerMatcherT,
) -> impl Matcher<ActualT = anyhow::Error> {
    ContextMatcher { inner }
}

struct ContextMatcher<InnerMatcherT: Matcher> {
    inner: InnerMatcherT,
}

impl<InnerMatcherT: Matcher<ActualT = String>> Matcher for ContextMatcher<InnerMatcherT> {
    type ActualT = anyhow::Error;

    fn matches(&self, actual: &Self::ActualT) -> MatcherResult {
        actual.chain().any(|e| self.inner.matches(&format!("{e}")).into()).into()
    }

    fn describe(&self, matcher_result: MatcherResult) -> Description {
        match matcher_result {
            MatcherResult::Match => {
                format!("has a context which {}", self.inner.describe(MatcherResult::Match)).into()
            }
            MatcherResult::NoMatch => format!(
                "doesn't have a context which {}",
                self.inner.describe(MatcherResult::Match)
            )
            .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use anyhow::anyhow;
    use googletest::prelude::*;
    use tonic::Code::{Internal, Unknown};

    #[googletest::test]
    fn code_matches_top_level_context() {
        expect_that!(anyhow!("message").context(Internal), code(Internal));
    }

    #[googletest::test]
    fn code_matches_nested_context() {
        expect_that!(
            anyhow!("message").context("context").context(Internal).context("extra context"),
            code(Internal)
        );
    }

    #[googletest::test]
    fn code_matches_updated_code() {
        expect_that!(anyhow!("message").context(Internal).context(Unknown), code(Unknown));
    }

    #[googletest::test]
    fn code_does_not_match_different_code() {
        expect_that!(anyhow!("message").context(Unknown), not(code(Internal)));
    }

    #[googletest::test]
    fn code_does_not_match_overwritten_code() {
        expect_that!(anyhow!("message").context(Internal).context(Unknown), not(code(Internal)));
    }

    #[googletest::test]
    fn code_does_not_match_no_code() {
        expect_that!(anyhow!("message"), not(code(Internal)));
    }

    #[googletest::test]
    fn has_context_matches_top_level_context() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("context2"))
        );
    }

    #[googletest::test]
    fn has_context_matches_intermediate_context() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("context1"))
        );
    }

    #[googletest::test]
    fn has_context_matches_root_cause() {
        expect_that!(
            anyhow!("message").context("context1").context("context2"),
            has_context(eq("message"))
        );
    }

    #[googletest::test]
    fn has_context_does_not_match_different_string() {
        expect_that!(anyhow!("message").context("context1"), not(has_context(eq("context2"))));
    }
}
