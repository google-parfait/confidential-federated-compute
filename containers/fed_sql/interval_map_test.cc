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

#include "containers/fed_sql/interval_map.h"

#include <cstdint>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

// Helper matcher for (Interval, Value) pairs in the map.
#define ENTRY(s, e, v) Pair(Interval<int>(s, e), v)

TEST(IntervalMapTest, EmptyMap) {
  IntervalMap<int, int> map;
  EXPECT_EQ(map.size(), 0);
  EXPECT_TRUE(map.empty());
  EXPECT_THAT(map, ElementsAre());
}

TEST(IntervalMapTest, OneInterval) {
  IntervalMap<int, int> map({{{1, 5}, 10}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(1, 5, 10)));
}

TEST(IntervalMapTest, MultipleIntervals) {
  IntervalMap<int, int> map({{{0, 10}, 4}, {{10, 20}, 3}, {{20, 30}, 2}});
  EXPECT_EQ(map.size(), 3);
  EXPECT_THAT(map,
              ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 3), ENTRY(20, 30, 2)));
}

TEST(IntervalMapTest, InitializationMergesAdjacentSameValue) {
  IntervalMap<int, int> map({{{0, 10}, 5}, {{10, 20}, 5}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 20, 5)));
}

TEST(IntervalMapTest, InitializationIgnoresEmptyIntervals) {
  IntervalMap<int, int> map({{{5, 5}, 1}, {{10, 20}, 3}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(10, 20, 3)));
}

TEST(IntervalMapTest, InitializationWithGaps) {
  IntervalMap<int, int> map({{{0, 5}, 1}, {{10, 15}, 2}});
  EXPECT_EQ(map.size(), 2);
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 5, 1), ENTRY(10, 15, 2)));
}

TEST(IntervalMapTest, Clear) {
  IntervalMap<int, int> map({{{0, 10}, 1}, {{10, 20}, 2}});
  map.Clear();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.size(), 0);
}

TEST(IntervalMapTest, Equals) {
  IntervalMap<int, int> map1({{{0, 10}, 1}, {{10, 20}, 2}});
  IntervalMap<int, int> map2({{{0, 10}, 1}, {{10, 20}, 2}});
  EXPECT_TRUE(map1 == map2);
  EXPECT_FALSE(map1 != map2);
}

TEST(IntervalMapTest, NotEqualsDifferentValues) {
  IntervalMap<int, int> map1({{{0, 10}, 1}});
  IntervalMap<int, int> map2({{{0, 10}, 2}});
  EXPECT_TRUE(map1 != map2);
  EXPECT_FALSE(map1 == map2);
}

TEST(IntervalMapTest, NotEqualsDifferentIntervals) {
  IntervalMap<int, int> map1({{{0, 10}, 1}});
  IntervalMap<int, int> map2({{{0, 5}, 1}});
  EXPECT_TRUE(map1 != map2);
}

TEST(IntervalMapTest, ForEachValueDecrementStoredIntervals) {
  IntervalMap<int, int> map({{{0, 10}, 4}, {{20, 30}, 3}, {{30, 40}, 2}});
  EXPECT_TRUE(map.ForEachValue({5, 30}, [](int& val) {
    --val;
    return true;
  }));
  EXPECT_THAT(map,
              ElementsAre(ENTRY(0, 5, 4), ENTRY(5, 10, 3), ENTRY(20, 40, 2)));
}

TEST(IntervalMapTest, ForEachValueNoStoredIntervalsInRange) {
  IntervalMap<int, int> map({{{0, 10}, 5}});
  EXPECT_TRUE(map.ForEachValue({30, 40}, [](int& val) {
    --val;
    return true;
  }));
  // No stored intervals in [30, 40), so map is unchanged.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 5)));
}

TEST(IntervalMapTest, ForEachValueEmptyInterval) {
  IntervalMap<int, int> map({{{0, 10}, 4}});
  EXPECT_TRUE(map.ForEachValue({5, 5}, [](int& val) {
    val = 0;
    return true;
  }));
  // No change since interval is empty.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4)));
}

TEST(IntervalMapTest, ForEachValueCheckPositive) {
  IntervalMap<int, int> map({{{0, 10}, 4}, {{10, 20}, 0}});
  // Returns false because [10, 20) has value 0.
  EXPECT_FALSE(map.ForEachValue({0, 20}, [](int& val) { return val > 0; }));
  // The map should be unchanged because the lambda didn't mutate.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 0)));
}

TEST(IntervalMapTest, ForEachValueCheckPositiveAllPositive) {
  IntervalMap<int, int> map({{{0, 10}, 4}, {{10, 20}, 3}});
  // Returns true because all stored values are > 0.
  EXPECT_TRUE(map.ForEachValue({0, 20}, [](int& val) { return val > 0; }));
  // The map should be unchanged.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 3)));
}

TEST(IntervalMapTest, ForEachValueDoubleValues) {
  IntervalMap<int, int> map({{{0, 10}, 4}, {{10, 20}, 3}});
  EXPECT_TRUE(map.ForEachValue({0, 20}, [](int& val) {
    val *= 2;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 8), ENTRY(10, 20, 6)));
}

TEST(IntervalMapTest, ForEachValueComplexSplitAndMerge) {
  // Start with a uniform map.
  IntervalMap<int, int> map({{{0, 100}, 10}});

  // Decrement the middle 3 times.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(map.ForEachValue({20, 40}, [](int& val) {
      --val;
      return true;
    }));
  }
  EXPECT_THAT(
      map, ElementsAre(ENTRY(0, 20, 10), ENTRY(20, 40, 7), ENTRY(40, 100, 10)));

  // Decrement an adjacent interval — triggers a merge on the left.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(map.ForEachValue({40, 60}, [](int& val) {
      --val;
      return true;
    }));
  }

  EXPECT_THAT(
      map, ElementsAre(ENTRY(0, 20, 10), ENTRY(20, 60, 7), ENTRY(60, 100, 10)));
}

TEST(IntervalMapTest, InsertEmptyInterval) {
  IntervalMap<int, int> map({{{0, 10}, 5}});
  EXPECT_TRUE(map.Insert({5, 5}, 99));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 5)));
}

TEST(IntervalMapTest, InsertIntoGap) {
  IntervalMap<int, int> map({{{5, 10}, 5}});
  EXPECT_TRUE(map.Insert({20, 30}, 5));
  EXPECT_TRUE(map.Insert({0, 5}, 3));
  EXPECT_THAT(map,
              ElementsAre(ENTRY(0, 5, 3), ENTRY(5, 10, 5), ENTRY(20, 30, 5)));
}

TEST(IntervalMapTest, InsertMergesBothSides) {
  IntervalMap<int, int> map({{{0, 10}, 7}, {{20, 30}, 7}});
  EXPECT_TRUE(map.Insert({10, 20}, 7));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 30, 7)));
}

TEST(IntervalMapTest, InsertOverlapReturnsFalse) {
  IntervalMap<int, int> map({{{0, 10}, 5}});
  EXPECT_FALSE(map.Insert({5, 15}, 3));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 5)));
}

TEST(IntervalMapTest, GetGapsEmptyMap) {
  IntervalMap<int, int> map;
  // The entire range is a gap.
  auto gaps = map.GetGaps({0, 100});
  EXPECT_THAT(gaps, ElementsAre(Interval<int>(0, 100)));
}

TEST(IntervalMapTest, GetGapsEmptyInterval) {
  IntervalMap<int, int> map({{{0, 10}, 5}});
  auto gaps = map.GetGaps({5, 5});
  EXPECT_TRUE(gaps.empty());
}

TEST(IntervalMapTest, GetGapsNoGaps) {
  IntervalMap<int, int> map({{{0, 20}, 5}});
  auto gaps = map.GetGaps({5, 15});
  EXPECT_TRUE(gaps.empty());
}

TEST(IntervalMapTest, GetGapsBetweenStoredIntervals) {
  IntervalMap<int, int> map({{{0, 10}, 1}, {{20, 30}, 2}, {{40, 50}, 3}});
  auto gaps = map.GetGaps({0, 50});
  EXPECT_THAT(gaps, ElementsAre(Interval<int>(10, 20), Interval<int>(30, 40)));
  gaps = map.GetGaps({5, 15});
  EXPECT_THAT(gaps, ElementsAre(Interval<int>(10, 15)));
}

TEST(IntervalMapTest, GetGapsBeyondStoredIntervals) {
  IntervalMap<int, int> map({{{10, 20}, 1}});
  auto gaps = map.GetGaps({0, 30});
  EXPECT_THAT(gaps, ElementsAre(Interval<int>(0, 10), Interval<int>(20, 30)));
}

TEST(IntervalMapTest, LastIntervalEndEmptyMap) {
  IntervalMap<int, int> map;
  EXPECT_FALSE(map.last_interval_end().has_value());
}

TEST(IntervalMapTest, LastIntervalEndWithIntervals) {
  IntervalMap<int, int> map({{{0, 10}, 1}, {{20, 30}, 2}, {{40, 100}, 3}});
  EXPECT_EQ(map.last_interval_end(), 100);
}

TEST(IntervalMapTest, EraseIfByStoredValue) {
  IntervalMap<int, int> map({{{0, 10}, 1}, {{10, 20}, 2}, {{20, 30}, 3}});
  map.EraseIf([](const auto& pair) { return pair.second < 3; });
  EXPECT_THAT(map, ElementsAre(ENTRY(20, 30, 3)));
}

TEST(IntervalMapTest, EraseIfByIntervalEnd) {
  IntervalMap<int, int> map(
      {{{0, 10}, 1}, {{10, 20}, 2}, {{20, 30}, 3}, {{30, 40}, 4}});
  map.EraseIf([](const auto& pair) { return pair.first.end() <= 20; });
  EXPECT_THAT(map, ElementsAre(ENTRY(20, 30, 3), ENTRY(30, 40, 4)));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
