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
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets { key: "baz" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  EXPECT_THAT(budget, UnorderedElementsAre(Pair("foo", 5), Pair("bar", 2),
                                           Pair("baz", 0)));
  auto keys = budget.GetKeys();
  EXPECT_THAT(keys.size(), 3);
  EXPECT_TRUE(keys.contains("foo"));
  EXPECT_TRUE(keys.contains("bar"));
  EXPECT_TRUE(keys.contains("baz"));
}

TEST(BudgetTest, Serialize) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets { key: "baz" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());

  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 5 }
        buckets { key: "bar" budget: 2 }
        buckets { key: "baz" budget: 0 }
        time_budget {}
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, StringParseAndSerialize) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets { key: "baz" budget: 1 }
        time_budget {}
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

TEST(BudgetTest, HasRemainingBudgetSimple) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
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
  // The default budget is zero, so no buckets should have any remaining budget.
  EXPECT_FALSE(budget2.HasRemainingBudget("foo"));
}

TEST(BudgetTest, HasRemainingBudgetWithInfiniteDefaultBudget) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "bar" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/std::nullopt);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // Should have budget in any bucket unless it is explicitly exhausted.
  EXPECT_TRUE(budget.HasRemainingBudget("foo"));
  EXPECT_FALSE(budget.HasRemainingBudget("bar"));
  // Should have remaining time budget.
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(100, 200)));
}

TEST(BudgetTest, UpdateDefaultBudget) {
  Budget budget(/*default_budget=*/5);
  RangeTracker range_tracker_1;
  range_tracker_1.AddKey("foo");
  range_tracker_1.AddKey("bar");
  EXPECT_TRUE(range_tracker_1.AddRange(1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker_1), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 4 }
                buckets { key: "bar" budget: 4 }
                time_budget {}
              )pb"));

  // Use a RangeTracker to update with different keys and expired keys.
  RangeTracker range_tracker_2;
  range_tracker_2.AddKey("baz");
  range_tracker_2.SetExpiredKeys({"bar"});
  EXPECT_TRUE(range_tracker_2.AddRange(5, 8));
  EXPECT_THAT(budget.UpdateBudget(range_tracker_2), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 4 }
                buckets { key: "baz" budget: 4 }
                time_budget {}
              )pb"));
}

TEST(BudgetTest, UpdateParsedBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 3 }
    buckets { key: "bar" budget: 2 }
    buckets { key: "expired_key1" budget: 2 }
  )pb");

  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // The set of buckets in the range tracker isn't the same.
  // Add some extra expired keys to the range tracker.
  RangeTracker range_tracker;
  range_tracker.SetExpiredKeys({"expired_key1", "expired_key2"});
  range_tracker.AddKey("bar");
  range_tracker.AddKey("baz");
  EXPECT_TRUE(range_tracker.AddRange(1, 5));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  // The "foo" bucket remains unchanged.
  // The "bar" bucket is updated.
  // The "baz" bucket is set to the default budget and then updated.
  // The "expired_key" bucket is removed.
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 3 }
                buckets { key: "bar" budget: 1 }
                buckets { key: "baz" budget: 4 }
                time_budget {}
              )pb"));
}

TEST(BudgetTest, UpdateBudgetConsumesOneUnitRegardlessOfRanges) {
  Budget budget(/*default_budget=*/5);
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  // Multiple disjoint ranges (e.g. autotuning and aggregation ranges merged
  // into a single release) consume exactly one unit of budget.
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_TRUE(range_tracker.AddRange(10, 14));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 4 }
                time_budget {}
              )pb"));

  // A subsequent update with a non-overlapping range consumes another unit.
  RangeTracker range_tracker_2;
  range_tracker_2.AddKey("foo");
  EXPECT_TRUE(range_tracker_2.AddRange(20, 30));
  EXPECT_THAT(budget.UpdateBudget(range_tracker_2), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 3 }
                time_budget {}
              )pb"));
}

TEST(BudgetTest, UpdateBudgetWithoutRangesFails) {
  Budget budget(/*default_budget=*/2);
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  EXPECT_THAT(budget.UpdateBudget(range_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(budget.begin(), budget.end());
}

TEST(BudgetTest, UpdateBudgetDecrementToZero) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 1 }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  EXPECT_TRUE(range_tracker.AddRange(3, 9));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  EXPECT_FALSE(budget.HasRemainingBudget("foo"));
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 0 }
                time_budget {}
              )pb"));
}

TEST(BudgetTest, UpdateExhaustedBudgetFails) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 3 }
        buckets { key: "bar" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  range_tracker.AddKey("bar");
  range_tracker.AddKey("baz");
  EXPECT_TRUE(range_tracker.AddRange(2, 3));
  EXPECT_THAT(budget.UpdateBudget(range_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  // The budget is left unmodified on failure.
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 3 }
                buckets { key: "bar" budget: 0 }
                time_budget {}
              )pb"));
}

TEST(BudgetTest, UpdateZeroDefaultBudgetFails) {
  Budget budget(/*default_budget=*/0);
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BudgetTest, UpdateInfiniteBudget) {
  Budget budget(/*default_budget=*/std::nullopt);
  RangeTracker range_tracker;
  range_tracker.AddKey("bar");
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  // The budget should remain empty - no "bar" bucket.
  EXPECT_EQ(budget.begin(), budget.end());
}

TEST(BudgetTest, UpdateBudgetWithTimeWindow) {
  Budget budget(/*default_budget=*/1);
  RangeTracker range_tracker;
  range_tracker.MergeAggWindow(Interval<uint64_t>(1200, 2400));

  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(1200, 2400)));
  // Update with time_window should update TimeBudget, not per-key budgets.
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Per-key budgets should be empty (no per-key update happened).
  EXPECT_EQ(budget.begin(), budget.end());

  // Time budget should be consumed.
  EXPECT_FALSE(budget.HasRemainingBudget(Interval<uint64_t>(1200, 2400)));
  // A window that was never consumed still has remaining budget.
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(3000, 3600)));
}

TEST(BudgetTest, UpdateBudgetWithTimeWindowCleansExpiredKeys) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" budget: 0 }
    buckets { key: "expired_key" budget: 0 }
  )pb");
  Budget budget(/*default_budget=*/1);
  EXPECT_THAT(budget.Parse(state), IsOk());

  RangeTracker range_tracker;
  range_tracker.MergeAggWindow(Interval<uint64_t>(1200, 2400));
  range_tracker.SetExpiredKeys({"expired_key"});

  // Update with time_window: should update time budget and clean expired keys.
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // expired_key should be gone, foo should remain unchanged.
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets { key: "foo" budget: 0 }
                time_budget {
                  anchor_time: 1200
                  intervals { count: 20 remaining_budget: 0 }
                }
              )pb"));
}

TEST(BudgetTest, SerializeAndParseTimeBudget) {
  Budget budget(/*default_budget=*/3);
  RangeTracker range_tracker;
  range_tracker.MergeAggWindow(Interval<uint64_t>(1200, 2400));
  EXPECT_TRUE(range_tracker.AddRange(1, 4));

  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Serialize and parse back.
  BudgetState serialized = budget.Serialize();
  Budget budget2(/*default_budget=*/3);
  EXPECT_THAT(budget2.Parse(serialized), IsOk());

  // Time budget state should survive the round-trip.
  EXPECT_TRUE(budget2.HasRemainingBudget(Interval<uint64_t>(1200, 2400)));
  EXPECT_THAT(budget2.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(serialized));
}

TEST(BudgetTest, UpdateBudgetRejectsPerKeyWhenTimeBudgetConsumed) {
  Budget budget(/*default_budget=*/3);

  RangeTracker time_tracker;
  time_tracker.MergeAggWindow(Interval<uint64_t>(1200, 2400));
  EXPECT_THAT(budget.UpdateBudget(time_tracker), IsOk());

  RangeTracker key_tracker;
  key_tracker.AddKey("foo");
  EXPECT_TRUE(key_tracker.AddRange(1, 4));
  EXPECT_THAT(budget.UpdateBudget(key_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Verify that updating with key_tracker also fails after parsing persisted
  // state.
  BudgetState serialized = budget.Serialize();
  Budget budget2(/*default_budget=*/3);
  EXPECT_THAT(budget2.Parse(serialized), IsOk());
  EXPECT_THAT(budget2.UpdateBudget(key_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
