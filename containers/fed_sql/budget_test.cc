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

#include "containers/fed_sql/budget.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(BudgetTest, Parse) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 7 }
    buckets { key: "bar" budget: 2 }
  )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // The "foo" bucket budget should be limited to 5.
  EXPECT_THAT(budget, UnorderedElementsAre(Pair("foo", 5), Pair("bar", 2)));
}

TEST(BudgetTest, Serialize) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 7 }
    buckets { key: "bar" budget: 2 }
  )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 5 }
                buckets { key: "bar" budget: 2 }
              )pb"));
}

TEST(BudgetTest, StringParseAndSerialize) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 7 }
    buckets { key: "bar" budget: 2 }
  )pb");
  std::string serialized_state = state.SerializeAsString();
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(serialized_state), IsOk());
  std::string new_serialized_state = budget.SerializeAsString();
  // The new serialized state may not be the same due to the repeated
  // field reordering, but the size should be the same.
  EXPECT_EQ(serialized_state.size(), new_serialized_state.size());
  // Make sure we can parse again
  EXPECT_THAT(budget.Parse(new_serialized_state), IsOk());
}

TEST(BudgetTest, StringParseBadInput) {
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse("foobar"), StatusIs(absl::StatusCode::kInternal));
}

TEST(BudgetTest, HasRemainingBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 7 }
    buckets { key: "bar" budget: 0 }
  )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  EXPECT_TRUE(budget.HasRemainingBudget("foo"));
  EXPECT_FALSE(budget.HasRemainingBudget("bar"));
  // This bucket is unknown so the default budget is assumed.
  EXPECT_TRUE(budget.HasRemainingBudget("foobar"));

  Budget budget2(/*default_budget=*/0);
  // The default budget is zero, so no buckets should have any
  // remaining budget.
  EXPECT_FALSE(budget2.HasRemainingBudget("foo"));
}

TEST(BudgetTest, HasRemainingBudgetWithInfiniteDefaultBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "bar" budget: 0 }
  )pb");
  Budget budget(/*default_budget=*/std::nullopt);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // Should have budget in any bucket unless it is explicitly exhausted.
  EXPECT_TRUE(budget.HasRemainingBudget("foo"));
  EXPECT_FALSE(budget.HasRemainingBudget("bar"));
}

TEST(BudgetTest, UpdateDefaultBudget) {
  Budget budget(/*default_budget=*/5);
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 4 }
                buckets { key: "bar" budget: 4 }
              )pb"));
}

TEST(BudgetTest, UpdateParsedBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 3 }
    buckets { key: "bar" budget: 2 }
  )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // The set of buckets in the range tracker isn't the same.
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("baz", 2, 3));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  // The first bucket remains unchanged, the second bucket is updated, and
  // the third bucket is set to the default budget and then updated.
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 3 }
                buckets { key: "bar" budget: 1 }
                buckets { key: "baz" budget: 4 }
              )pb"));
}

TEST(BudgetTest, UpdateExhaustedBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 3 }
    buckets { key: "bar" budget: 0 }
  )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("baz", 2, 3));
  EXPECT_THAT(budget.UpdateBudget(range_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BudgetTest, UpdateInfiniteBudget) {
  Budget budget(/*default_budget=*/std::nullopt);
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  // The budget should remain empty - no "bar" bucket.
  EXPECT_EQ(budget.begin(), budget.end());
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
