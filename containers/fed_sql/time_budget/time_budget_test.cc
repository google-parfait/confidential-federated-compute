// Copyright 2026 Google LLC.
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

#include "containers/fed_sql/time_budget/time_budget.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/interval.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Eq;

TEST(TimeBudgetTest, UnlimitedBudget) {
  TimeBudget budget(std::nullopt);
  EXPECT_TRUE(budget.HasUnlimitedBudget());

  // HasRemainingBudget should always be true
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(0, 1000)));
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(5000, 6000)));

  // UpdateBudget is a no-op and should return ok
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(100, 200)), IsOk());
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(100, 200)));
}

TEST(TimeBudgetTest, EmptyTimeWindow) {
  TimeBudget budget(1);
  EXPECT_FALSE(budget.HasUnlimitedBudget());

  // Empty window is always true
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(120, 120)));
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(120, 120)), IsOk());
  EXPECT_TRUE(budget.HasRemainingBudget(Interval<uint64_t>(120, 120)));
}

TEST(TimeBudgetTest, HasRemainingBudgetBeforeAnchor) {
  TimeBudget budget(5);
  // First update sets anchor (1200 is minute-aligned)
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(1200, 2400)), IsOk());
  EXPECT_EQ(budget.anchor_time(), 20);  // 1200 seconds = 20 minutes

  // Checking time window starting before anchor should return false
  EXPECT_FALSE(budget.HasRemainingBudget(Interval<uint64_t>(600, 1800)));
  EXPECT_FALSE(budget.HasRemainingBudget(Interval<uint64_t>(0, 60)));
}

TEST(TimeBudgetTest, UpdateBudgetBeforeAnchorFails) {
  TimeBudget budget(5);
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(1200, 2400)), IsOk());

  // Updating window starting before anchor should return error
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(600, 1800)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(TimeBudgetTest, UpdateBudgetSuccess) {
  TimeBudget budget(1);
  uint64_t anchor = 12000;  // minute-aligned

  // First update sets the anchor and consumes budget for one minute.
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(anchor, anchor + 60)),
              IsOk());
  EXPECT_FALSE(
      budget.HasRemainingBudget(Interval<uint64_t>(anchor, anchor + 60)));

  // Update a window with fractional minutes: anchor + 120s to anchor + 181s
  // Smallest enclosing complete minute buckets since epoch:
  // start_aligned = anchor + 120
  // end_aligned = anchor + 240
  EXPECT_THAT(
      budget.UpdateBudget(Interval<uint64_t>(anchor + 120, anchor + 181)),
      IsOk());

  // Every point in the window [anchor + 120, anchor + 240) should now have 0
  // budget.
  EXPECT_FALSE(budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 120, anchor + 240)));
  EXPECT_FALSE(budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 120, anchor + 181)));

  // Other windows should still have budget
  EXPECT_TRUE(
      budget.HasRemainingBudget(Interval<uint64_t>(anchor + 60, anchor + 120)));
}

TEST(TimeBudgetTest, SerializeAndParse) {
  TimeBudget budget(1);
  uint64_t anchor = 60000;

  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(anchor, anchor + 60)),
              IsOk());
  EXPECT_THAT(
      budget.UpdateBudget(Interval<uint64_t>(anchor + 125, anchor + 185)),
      IsOk());

  BudgetState::TimeBudgetState serialized = budget.Serialize();
  EXPECT_EQ(serialized.anchor_time(), anchor);  // Serialized as seconds
  EXPECT_EQ(serialized.intervals_size(), 2);
  EXPECT_EQ(serialized.intervals(0).start_index(), 0);
  EXPECT_EQ(serialized.intervals(0).count(), 1);
  EXPECT_EQ(serialized.intervals(0).remaining_budget(), 0);
  EXPECT_EQ(serialized.intervals(1).start_index(), 2);
  EXPECT_EQ(serialized.intervals(1).count(), 2);
  EXPECT_EQ(serialized.intervals(1).remaining_budget(), 0);

  TimeBudget new_budget(1);
  EXPECT_THAT(new_budget.Parse(serialized), IsOk());

  EXPECT_EQ(new_budget.anchor_time(), anchor / 60);  // anchor_time() is minutes
  EXPECT_FALSE(
      new_budget.HasRemainingBudget(Interval<uint64_t>(anchor, anchor + 60)));
  EXPECT_FALSE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 120, anchor + 240)));
  EXPECT_FALSE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 125, anchor + 185)));
  EXPECT_TRUE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 240, anchor + 300)));
}

TEST(TimeBudgetTest, SerializeAndParseAsString) {
  TimeBudget budget(1);
  uint64_t anchor = 60000;

  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(anchor, anchor + 60)),
              IsOk());
  EXPECT_THAT(
      budget.UpdateBudget(Interval<uint64_t>(anchor + 125, anchor + 185)),
      IsOk());

  std::string serialized = budget.SerializeAsString();

  TimeBudget new_budget(1);
  EXPECT_THAT(new_budget.Parse(serialized), IsOk());

  EXPECT_EQ(new_budget.anchor_time(), anchor / 60);  // anchor_time() is minutes
  EXPECT_FALSE(
      new_budget.HasRemainingBudget(Interval<uint64_t>(anchor, anchor + 60)));
  EXPECT_FALSE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 120, anchor + 240)));
  EXPECT_FALSE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 125, anchor + 185)));
  EXPECT_TRUE(new_budget.HasRemainingBudget(
      Interval<uint64_t>(anchor + 240, anchor + 300)));
}

TEST(TimeBudgetTest, ParseCapsMaxBudget) {
  TimeBudget budget(1);
  BudgetState::TimeBudgetState proto_state;
  proto_state.set_anchor_time(12000);

  // Add interval with remaining_budget greater than default
  auto* int1 = proto_state.add_intervals();
  int1->set_start_index(0);
  int1->set_count(10);
  int1->set_remaining_budget(5);

  EXPECT_THAT(budget.Parse(proto_state), IsOk());

  // Verify remaining_budget is capped at default budget.
  EXPECT_THAT(budget.UpdateBudget(Interval<uint64_t>(12000, 12600)), IsOk());
  EXPECT_FALSE(budget.HasRemainingBudget(Interval<uint64_t>(12000, 12600)));
}

TEST(TimeBudgetTest, ParseOverlappingIntervals) {
  // Parsing a state with overlapping intervals should return error.
  BudgetState::TimeBudgetState invalid_state;
  invalid_state.set_anchor_time(12000);
  auto* i1 = invalid_state.add_intervals();
  i1->set_start_index(0);
  i1->set_count(10);
  i1->set_remaining_budget(1);

  auto* i2 = invalid_state.add_intervals();
  i2->set_start_index(5);
  i2->set_count(10);
  i2->set_remaining_budget(1);

  TimeBudget invalid_budget(3);
  EXPECT_THAT(invalid_budget.Parse(invalid_state),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
