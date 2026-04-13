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
  EXPECT_EQ(map.default_value(), 0);
}

TEST(IntervalMapTest, EmptyMapWithDefault) {
  IntervalMap<int, int> map(42);
  EXPECT_EQ(map.size(), 0);
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.default_value(), 42);
}

TEST(IntervalMapTest, OneInterval) {
  IntervalMap<int, int> map(20, {{{1, 5}, 10}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(1, 5, 10)));
}

TEST(IntervalMapTest, MultipleIntervals) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}, {{10, 20}, 3}, {{20, 30}, 2}});
  EXPECT_EQ(map.size(), 3);
  EXPECT_THAT(map,
              ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 3), ENTRY(20, 30, 2)));
}

TEST(IntervalMapTest, InitializationMergesAdjacentSameValue) {
  IntervalMap<int, int> map(10, {{{0, 10}, 5}, {{10, 20}, 5}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 20, 5)));
}

TEST(IntervalMapTest, InitializationIgnoresEmptyIntervals) {
  IntervalMap<int, int> map(5, {{{5, 5}, 1}, {{10, 20}, 3}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(10, 20, 3)));
}

TEST(IntervalMapTest, InitializationIgnoresDefaultValuedIntervals) {
  IntervalMap<int, int> map(5, {{{0, 10}, 5}, {{10, 20}, 3}});
  EXPECT_EQ(map.size(), 1);
  EXPECT_THAT(map, ElementsAre(ENTRY(10, 20, 3)));
}

TEST(IntervalMapTest, InitializationWithGaps) {
  IntervalMap<int, int> map(5, {{{0, 5}, 1}, {{10, 15}, 2}});
  EXPECT_EQ(map.size(), 2);
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 5, 1), ENTRY(10, 15, 2)));
}

TEST(IntervalMapTest, Clear) {
  IntervalMap<int, int> map(10, {{{0, 10}, 1}, {{10, 20}, 2}});
  map.Clear();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.size(), 0);
  // default_value should be preserved after clear.
  EXPECT_EQ(map.default_value(), 10);
}

TEST(IntervalMapTest, Equals) {
  IntervalMap<int, int> map1(10, {{{0, 10}, 1}, {{10, 20}, 2}});
  IntervalMap<int, int> map2(10, {{{0, 10}, 1}, {{10, 20}, 2}});
  EXPECT_TRUE(map1 == map2);
  EXPECT_FALSE(map1 != map2);
}

TEST(IntervalMapTest, NotEqualsDifferentValues) {
  IntervalMap<int, int> map1(10, {{{0, 10}, 1}});
  IntervalMap<int, int> map2(10, {{{0, 10}, 2}});
  EXPECT_TRUE(map1 != map2);
  EXPECT_FALSE(map1 == map2);
}

TEST(IntervalMapTest, NotEqualsDifferentIntervals) {
  IntervalMap<int, int> map1(10, {{{0, 10}, 1}});
  IntervalMap<int, int> map2(10, {{{0, 5}, 1}});
  EXPECT_TRUE(map1 != map2);
}

TEST(IntervalMapTest, NotEqualsDifferentDefault) {
  IntervalMap<int, int> map1(0);
  IntervalMap<int, int> map2(1);
  EXPECT_TRUE(map1 != map2);
  EXPECT_FALSE(map1 == map2);
}

TEST(IntervalMapTest, ForEachValueDecrementOverlap) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}, {{20, 30}, 3}, {{30, 40}, 2}});
  EXPECT_TRUE(map.ForEachValue({5, 30}, [](int& val) {
    --val;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 5, 4), ENTRY(5, 10, 3),
                               ENTRY(10, 20, 9), ENTRY(20, 40, 2)));
}

TEST(IntervalMapTest, ForEachValueDecrementNoOverlap) {
  IntervalMap<int, int> map(10, {{{0, 10}, 5}});
  EXPECT_TRUE(map.ForEachValue({30, 40}, [](int& val) {
    --val;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 5), ENTRY(30, 40, 9)));
}

TEST(IntervalMapTest, ForEachValueEmptyInterval) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}});
  EXPECT_TRUE(map.ForEachValue({5, 5}, [](int& val) {
    val = 0;
    return true;
  }));
  // No change since interval is empty.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4)));
}

TEST(IntervalMapTest, ForEachValueCheckPositive) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}, {{10, 20}, 0}});
  // Returns false because [10, 20) has value 0.
  EXPECT_FALSE(map.ForEachValue({0, 20}, [](int& val) { return val > 0; }));
  // The map should be unchanged because the lambda didn't mutate.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 0)));
}

TEST(IntervalMapTest, ForEachValueCheckPositiveAllPositive) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}, {{10, 20}, 3}});
  // Returns true because all values are > 0.
  EXPECT_TRUE(map.ForEachValue({0, 20}, [](int& val) { return val > 0; }));
  // The map should be unchanged.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 4), ENTRY(10, 20, 3)));
}

TEST(IntervalMapTest, ForEachValueCheckPositiveInGap) {
  // Entirely in default gap — default is 10, so all positive.
  IntervalMap<int, int> map(10, {{{0, 5}, 3}});
  EXPECT_TRUE(map.ForEachValue({10, 20}, [](int& val) { return val > 0; }));
  // Default gaps materialized and cleaned up — map should be unchanged.
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 5, 3)));
}

TEST(IntervalMapTest, ForEachValueSetToDefault) {
  // When ForEachValue sets a value to the default, it should be cleaned up.
  IntervalMap<int, int> map(10, {{{0, 20}, 5}});
  EXPECT_TRUE(map.ForEachValue({0, 10}, [](int& val) {
    val = 10;  // Set to default value.
    return true;
  }));
  // [0, 10) should be removed (it's now the default).
  EXPECT_THAT(map, ElementsAre(ENTRY(10, 20, 5)));
}

TEST(IntervalMapTest, ForEachValueDoubleValues) {
  IntervalMap<int, int> map(10, {{{0, 10}, 4}, {{10, 20}, 3}});
  EXPECT_TRUE(map.ForEachValue({0, 20}, [](int& val) {
    val *= 2;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 10, 8), ENTRY(10, 20, 6)));
}

TEST(IntervalMapTest, ForEachValueMutateInGap) {
  // Apply over a range that is entirely in the default gap.
  IntervalMap<int, int> map(10, {{{0, 5}, 3}});
  EXPECT_TRUE(map.ForEachValue({5, 15}, [](int& val) {
    val -= 2;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 5, 3), ENTRY(5, 15, 8)));
}

TEST(IntervalMapTest, ForEachValueComplexSplitAndMerge) {
  // Start with a uniform map.
  IntervalMap<int, int> map(20, {{{0, 100}, 10}});

  // Decrement the middle 3 times.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(map.ForEachValue({20, 40}, [](int& val) {
      --val;
      return true;
    }));
  }
  EXPECT_THAT(
      map, ElementsAre(ENTRY(0, 20, 10), ENTRY(20, 40, 7), ENTRY(40, 100, 10)));

  // Decrement an interval that causes a merge on the right.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(map.ForEachValue({40, 60}, [](int& val) {
      --val;
      return true;
    }));
  }
  EXPECT_THAT(
      map, ElementsAre(ENTRY(0, 20, 10), ENTRY(20, 60, 7), ENTRY(60, 100, 10)));
}

TEST(IntervalMapTest, ForEachValueMergesInternalIntervals) {
  // Map has gaps: [0,5)→3, gap [5,15), [15,20)→7, default=10.
  // Setting all to 5 should merge into a single [0,20)→5.
  IntervalMap<int, int> map(10, {{{0, 5}, 3}, {{15, 20}, 7}});
  EXPECT_TRUE(map.ForEachValue({0, 20}, [](int& val) {
    val = 5;
    return true;
  }));
  EXPECT_THAT(map, ElementsAre(ENTRY(0, 20, 5)));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
