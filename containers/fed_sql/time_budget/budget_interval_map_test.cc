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

#include "containers/fed_sql/time_budget/budget_interval_map.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

TEST(BudgetIntervalMapTest, EmptyMap) {
  BudgetIntervalMap map(5);
  EXPECT_TRUE(map.HasBudget({0, 100}));
  // Empty interval is always true
  EXPECT_TRUE(map.HasBudget({0, 0}));
}

TEST(BudgetIntervalMapTest, Equality) {
  BudgetIntervalMap map1(5);
  BudgetIntervalMap map2(5);
  EXPECT_EQ(map1, map2);
  EXPECT_TRUE(map1.SubtractBudget({0, 10}));
  EXPECT_TRUE(map2.SubtractBudget({0, 10}));
  EXPECT_EQ(map1, map2);
}

TEST(BudgetIntervalMapTest, InequalityDifferentDefault) {
  BudgetIntervalMap map1(5);
  BudgetIntervalMap map2(10);
  EXPECT_TRUE(map1 != map2);
  EXPECT_FALSE(map1 == map2);
}

TEST(BudgetIntervalMapTest, InequalityDifferentStoredIntervals) {
  BudgetIntervalMap map1(5);
  BudgetIntervalMap map2(5);
  EXPECT_TRUE(map1.SubtractBudget({0, 10}));
  EXPECT_TRUE(map1 != map2);
}

TEST(BudgetIntervalMapTest, SubtractAndHasBudget) {
  BudgetIntervalMap map(2);
  EXPECT_TRUE(map.SubtractBudget({0, 10}));
  EXPECT_TRUE(map.HasBudget({0, 10}));
  EXPECT_TRUE(map.HasBudget({10, 20}));
  EXPECT_TRUE(map.SubtractBudget({0, 10}));
  EXPECT_FALSE(map.HasBudget({0, 10}));
  // Crossing the exhausted range should also fail.
  EXPECT_FALSE(map.HasBudget({5, 15}));
  EXPECT_TRUE(map.HasBudget({10, 20}));
  // Subtracting budget again fails since it's already 0.
  EXPECT_FALSE(map.SubtractBudget({0, 10}));
}

TEST(BudgetIntervalMapTest, SubtractBudgetEmptyInterval) {
  BudgetIntervalMap map(5);
  BudgetIntervalMap empty(5);
  // Empty — no-op
  EXPECT_TRUE(map.SubtractBudget({0, 0}));
  EXPECT_EQ(map, empty);
}

TEST(BudgetIntervalMapTest, SubtractBudgetMergesIntervals) {
  BudgetIntervalMap map(5);
  EXPECT_TRUE(map.SubtractBudget({0, 10}));
  EXPECT_TRUE(map.SubtractBudget({10, 20}));
  BudgetIntervalMap expected(5);
  EXPECT_TRUE(expected.SubtractBudget({0, 20}));
  EXPECT_EQ(map, expected);
}

TEST(BudgetIntervalMapTest, SubtractBudgetComplexScenario) {
  BudgetIntervalMap map(3);  // default budget = 3

  // [0, 30) -> 2
  EXPECT_TRUE(map.SubtractBudget({0, 30}));
  // [0,10) -> 2, [10, 20) -> 1, [20, 30) -> 2
  EXPECT_TRUE(map.SubtractBudget({10, 20}));
  // [0,20) -> 1, [20, 30) -> 2
  EXPECT_TRUE(map.SubtractBudget({0, 10}));

  // All remaining budget > 0.
  EXPECT_TRUE(map.HasBudget({0, 30}));
  EXPECT_TRUE(map.HasBudget({30, 50}));

  // Exhaust [0, 20).
  EXPECT_TRUE(map.SubtractBudget({0, 20}));
  EXPECT_FALSE(map.HasBudget({0, 20}));
  EXPECT_TRUE(map.HasBudget({20, 30}));
  EXPECT_TRUE(map.HasBudget({30, 100}));
}

TEST(BudgetIntervalMapTest, InsertInterval) {
  BudgetIntervalMap map(5);
  EXPECT_TRUE(map.Insert({0, 10}, 3));
  EXPECT_TRUE(map.HasBudget({0, 10}));

  // Inserting overlap fails
  EXPECT_FALSE(map.Insert({5, 15}, 2));
  // Inserting value > default_budget is not allowed.
  EXPECT_FALSE(map.Insert({10, 20}, 10));

  // Inserting empty interval.
  EXPECT_TRUE(map.Insert({15, 15}, 3));
  EXPECT_TRUE(map.HasBudget({15, 15}));
  // Inserting value = default_budget.
  EXPECT_TRUE(map.Insert({10, 15}, 5));
  EXPECT_TRUE(map.HasBudget({10, 15}));
}

TEST(BudgetIntervalMapTest, CleanupStaleIntervalsEmptyMap) {
  BudgetIntervalMap map(5);
  // Should be a no-op on an empty map.
  map.CleanupStaleIntervals(100);
  EXPECT_TRUE(map.empty());
}

TEST(BudgetIntervalMapTest, CleanupStaleIntervalsRemovesExpired) {
  BudgetIntervalMap map(5);
  map.Insert({0, 10}, 3);
  map.Insert({10, 20}, 2);
  map.Insert({40, 70}, 2);
  map.Insert({100, 110}, 4);

  // TTL = 50. Latest interval ends at 110.
  // expiration_cutoff = 110 - 50 = 60.
  // Intervals [0,10) and [10,20) end before 60 -> removed.
  map.CleanupStaleIntervals(50);

  BudgetIntervalMap expected(5);
  expected.Insert({40, 70}, 2);
  expected.Insert({100, 110}, 4);
  EXPECT_EQ(map, expected);
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
