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

BudgetInfo CreateBudgetInfo(
    uint32_t budget, std::optional<Interval<uint64_t>> range = std::nullopt) {
  BudgetInfo info(budget);
  info.consumed_range = range;
  return info;
}

TEST(BudgetTest, Parse) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets {
          key: "partial"
          budget: 0
          consumed_range_start: 10
          consumed_range_end: 20
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  EXPECT_THAT(
      budget,
      UnorderedElementsAre(
          Pair("foo", CreateBudgetInfo(5)), Pair("bar", CreateBudgetInfo(2)),
          Pair("partial", CreateBudgetInfo(0, Interval<uint64_t>(10, 20)))));
}

TEST(BudgetTest, Serialize) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets {
          key: "partial"
          budget: 1
          consumed_range_start: 10
          consumed_range_end: 20
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());

  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 5 }
        buckets { key: "bar" budget: 2 }
        buckets {
          key: "partial"
          budget: 1
          consumed_range_start: 10
          consumed_range_end: 20
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, StringParseAndSerialize) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 7 }
        buckets { key: "bar" budget: 2 }
        buckets {
          key: "partial"
          budget: 1
          consumed_range_start: 10
          consumed_range_end: 20
        }
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
  EXPECT_TRUE(budget.HasRemainingBudget("foo", 100));
  EXPECT_FALSE(budget.HasRemainingBudget("bar", 100));
  // This bucket is unknown so the default budget is assumed.
  EXPECT_TRUE(budget.HasRemainingBudget("foobar", 100));

  Budget budget2(/*default_budget=*/0);
  // The default budget is zero, so no buckets should have any remaining budget.
  EXPECT_FALSE(budget2.HasRemainingBudget("foo", 100));
}

TEST(BudgetTest, HasRemainingBudgetWithPartialRange) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "a" budget: 2 }
        buckets {
          key: "b"
          budget: 0
          consumed_range_start: 10
          consumed_range_end: 20
        }
        buckets { key: "c" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());

  EXPECT_TRUE(budget.HasRemainingBudget("a", 5));    // budget > 1
  EXPECT_TRUE(budget.HasRemainingBudget("b", 5));    // Outside range [10, 20)
  EXPECT_FALSE(budget.HasRemainingBudget("b", 10));  // Inside range
  EXPECT_FALSE(budget.HasRemainingBudget("b", 19));  // Inside range
  EXPECT_TRUE(budget.HasRemainingBudget("b", 20));   // Outside range

  EXPECT_FALSE(budget.HasRemainingBudget("c", 100));  // budget 0
}

TEST(BudgetTest, HasRemainingBudgetWithInfiniteDefaultBudget) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "bar" budget: 0 }
      )pb");
  Budget budget(/*default_budget=*/std::nullopt);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // Should have budget in any bucket unless it is explicitly exhausted.
  EXPECT_TRUE(budget.HasRemainingBudget("foo", 100));
  EXPECT_FALSE(budget.HasRemainingBudget("bar", 100));
}

TEST(BudgetTest, UpdateDefaultBudget) {
  Budget budget(/*default_budget=*/5);
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets {
                  key: "foo"
                  budget: 4
                  consumed_range_start: 1
                  consumed_range_end: 4
                }
                buckets {
                  key: "bar"
                  budget: 4
                  consumed_range_start: 1
                  consumed_range_end: 4
                }
              )pb"));
}

TEST(BudgetTest, UpdateParsedBudget) {
  BudgetState state = PARSE_TEXT_PROTO(R"pb(
    buckets {
      key: "foo"
      budget: 3
      consumed_range_start: 1
      consumed_range_end: 4
    }
    buckets { key: "bar" budget: 2 }
  )pb");

  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  // The set of buckets in the range tracker isn't the same.
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 5));
  EXPECT_TRUE(range_tracker.AddRange("baz", 2, 3));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());
  // The first bucket remains unchanged, the second bucket is updated, and
  // the third bucket is set to the default budget and then updated.
  EXPECT_THAT(budget.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets {
                  key: "foo"
                  budget: 3
                  consumed_range_start: 1
                  consumed_range_end: 4
                }
                buckets {
                  key: "bar"
                  budget: 1
                  consumed_range_start: 1
                  consumed_range_end: 5
                }
                buckets {
                  key: "baz"
                  budget: 4
                  consumed_range_start: 2
                  consumed_range_end: 3
                }
              )pb"));
}

TEST(BudgetTest, UpdateBudgetNewRanges) {
  Budget budget(/*default_budget=*/5);
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("bar", 10, 14));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 4
          consumed_range_start: 1
          consumed_range_end: 4
        }
        buckets {
          key: "bar"
          budget: 4
          consumed_range_start: 10
          consumed_range_end: 14
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateParsedBudgetExistingKeysNoOverlap) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 3
          consumed_range_start: 1
          consumed_range_end: 4
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 10, 14));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget not decremented, consumed range extended
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 3
          consumed_range_start: 1
          consumed_range_end: 14
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateParsedBudgetExistingKeysOverlap) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 3
          consumed_range_start: 1
          consumed_range_end: 10
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 5, 14));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget decremented, new range is the current range
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 2
          consumed_range_start: 5
          consumed_range_end: 14
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateParsedBudgetOldRangeNotContained) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 3
          consumed_range_start: 5
          consumed_range_end: 10
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  // New range is not contained in the old range.
  EXPECT_TRUE(range_tracker.AddRange("foo", 1, 5));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget range expanded.
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 3
          consumed_range_start: 1
          consumed_range_end: 10
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateBudgetDecrementToZero) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 1
          consumed_range_start: 1
          consumed_range_end: 10
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("foo", 3, 9));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget zero, partial range is updated.
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 0
          consumed_range_start: 3
          consumed_range_end: 9
        }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateBudgetZeroConsumed) {
  BudgetState initial_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets {
          key: "foo"
          budget: 0
          consumed_range_start: 0
          consumed_range_end: 10
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(initial_state), IsOk());

  RangeTracker range_tracker;
  EXPECT_TRUE(
      range_tracker.AddRange("foo", 11, std::numeric_limits<uint64_t>::max()));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget zero, partial range is cleared.
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 0 }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateBudgetFullRange) {
  Budget budget(/*default_budget=*/2);
  RangeTracker range_tracker;
  EXPECT_TRUE(
      range_tracker.AddRange("foo", 0, std::numeric_limits<uint64_t>::max()));
  EXPECT_THAT(budget.UpdateBudget(range_tracker), IsOk());

  // Budget decremented, range fields are not set because it's a full range
  BudgetState expected_state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 1 }
      )pb");
  EXPECT_THAT(budget.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(expected_state));
}

TEST(BudgetTest, UpdateExhaustedBudgetPartialRange) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
        buckets { key: "foo" budget: 3 }
        buckets {
          key: "bar"
          budget: 0
          consumed_range_start: 1
          consumed_range_end: 10
        }
      )pb");
  Budget budget(/*default_budget=*/5);
  EXPECT_THAT(budget.Parse(state), IsOk());
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange("bar", 1, 4));
  EXPECT_TRUE(range_tracker.AddRange("baz", 2, 3));
  EXPECT_THAT(budget.UpdateBudget(range_tracker),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BudgetTest, UpdateExhaustedBudgetNoPartialRange) {
  BudgetState state = PARSE_TEXT_PROTO(
      R"pb(
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
