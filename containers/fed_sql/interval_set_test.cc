// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "containers/fed_sql/interval_set.h"

#include <initializer_list>
#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::testing::ElementsAre;

constexpr int kMaxValue = std::numeric_limits<int>::max();

TEST(IntervalSetTest, EmptySet) {
  IntervalSet<int> set;
  EXPECT_EQ(set.size(), 0);
  EXPECT_THAT(set, ElementsAre());
  EXPECT_FALSE(set.Contains(0));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 0));
  EXPECT_TRUE(set.BoundingInterval().empty());
}

TEST(IntervalSetTest, OneInterval) {
  IntervalSet<int> set({{1, 5}});
  EXPECT_EQ(set.size(), 1);
  EXPECT_THAT(set, ElementsAre(Interval(1, 5)));
  EXPECT_FALSE(set.Contains(0));
  EXPECT_FALSE(set.Contains(5));
  EXPECT_FALSE(set.Contains(6));
  EXPECT_TRUE(set.Contains(1));
  EXPECT_TRUE(set.Contains(4));
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 5));
}

TEST(IntervalSetTest, MaxValueIntervalEnd) {
  IntervalSet<int> set({{1, kMaxValue}});
  EXPECT_FALSE(set.Contains(0));
  EXPECT_TRUE(set.Contains(kMaxValue));
  EXPECT_EQ(set.BoundingInterval(), Interval(1, kMaxValue));
}

TEST(IntervalSetTest, Assign) {
  IntervalSet<int> set({{1, 5}});
  std::initializer_list<Interval<int>> intervals = {{0, 1}, {2, 3}};
  EXPECT_TRUE(set.Assign(intervals.begin(), intervals.end()));
  EXPECT_THAT(set, ElementsAre(Interval(0, 1), Interval(2, 3)));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 3));
}

TEST(IntervalSetTest, AssignUnordered) {
  IntervalSet<int> set({{1, 5}});
  std::initializer_list<Interval<int>> intervals = {{2, 3}, {0, 1}};
  EXPECT_FALSE(set.Assign(intervals.begin(), intervals.end()));
  EXPECT_THAT(set, ElementsAre(Interval(1, 5)));  // Remains unchanged
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 5));
}

TEST(IntervalSetTest, AssignNotDisjoint) {
  IntervalSet<int> set({{1, 5}});
  std::initializer_list<Interval<int>> intervals = {{0, 3}, {3, 4}};
  EXPECT_FALSE(set.Assign(intervals.begin(), intervals.end()));
  EXPECT_THAT(set, ElementsAre(Interval<int>(1, 5)));  // Remains unchanged
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 5));
}

TEST(IntervalSetTest, TwoIntervals) {
  IntervalSet<int> set({{1, 3}, {6, 9}});
  EXPECT_EQ(set.size(), 2);
  EXPECT_THAT(set, ElementsAre(Interval(1, 3), Interval(6, 9)));
  EXPECT_FALSE(set.Contains(0));
  EXPECT_FALSE(set.Contains(5));
  EXPECT_FALSE(set.Contains(10));
  EXPECT_TRUE(set.Contains(2));
  EXPECT_TRUE(set.Contains(7));
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 9));
}

TEST(IntervalSetTest, EmptyIntervalsIgnored) {
  IntervalSet<int> set;
  EXPECT_TRUE(set.Add(Interval<int>()));
  EXPECT_TRUE(set.Add({10, 10}));
  EXPECT_THAT(set, ElementsAre());
  EXPECT_TRUE(set.BoundingInterval().empty());
}

TEST(IntervalSetTest, AddMultiple) {
  IntervalSet<int> set;
  EXPECT_TRUE(set.Add({3, 4}));
  EXPECT_EQ(set.BoundingInterval(), Interval(3, 4));
  EXPECT_TRUE(set.Add({5, 6}));
  EXPECT_EQ(set.BoundingInterval(), Interval(3, 6));
  EXPECT_TRUE(set.Add({11, 12}));
  EXPECT_EQ(set.BoundingInterval(), Interval(3, 12));
  EXPECT_THAT(set,
              ElementsAre(Interval(3, 4), Interval(5, 6), Interval(11, 12)));

  EXPECT_TRUE(set.Add({1, 2}));
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 12));
  EXPECT_THAT(set, ElementsAre(Interval(1, 2), Interval(3, 4), Interval(5, 6),
                               Interval(11, 12)));

  EXPECT_TRUE(set.Add({0, 1}));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 12));
  EXPECT_THAT(set, ElementsAre(Interval(0, 2), Interval(3, 4), Interval(5, 6),
                               Interval(11, 12)));

  EXPECT_TRUE(set.Add({8, 9}));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 12));

  EXPECT_TRUE(set.Add({6, 7}));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 12));
  EXPECT_THAT(set, ElementsAre(Interval(0, 2), Interval(3, 4), Interval(5, 7),
                               Interval(8, 9), Interval(11, 12)));

  EXPECT_TRUE(set.Add({12, 15}));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 15));

  EXPECT_TRUE(set.Add({10, 11}));

  EXPECT_TRUE(set.Add({2, 3}));
  EXPECT_THAT(set, ElementsAre(Interval(0, 4), Interval(5, 7), Interval(8, 9),
                               Interval(10, 15)));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 15));

  EXPECT_TRUE(set.Add({7, 8}));
  EXPECT_THAT(set,
              ElementsAre(Interval(0, 4), Interval(5, 9), Interval(10, 15)));

  EXPECT_TRUE(set.Add({9, 10}));
  EXPECT_THAT(set, ElementsAre(Interval(0, 4), Interval(5, 15)));

  EXPECT_TRUE(set.Add({4, 5}));
  EXPECT_EQ(set.BoundingInterval(), Interval(0, 15));
  EXPECT_THAT(set, ElementsAre(Interval(0, 15)));
}

TEST(IntervalSetTest, AddOverlapping) {
  IntervalSet<int> set({{1, 3}, {6, 9}});
  EXPECT_FALSE(set.Add({0, 2}));
  EXPECT_FALSE(set.Add({0, 4}));
  EXPECT_FALSE(set.Add({2, 6}));
  EXPECT_FALSE(set.Add({3, 7}));
  EXPECT_FALSE(set.Add({6, 9}));
  EXPECT_FALSE(set.Add({7, 9}));
  EXPECT_FALSE(set.Add({8, 10}));
  EXPECT_THAT(set, ElementsAre(Interval(1, 3), Interval(6, 9)));
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 9));
}

TEST(IntervalSetTest, Merge) {
  IntervalSet<int> set1(
      {{2, 4}, {6, 8}, {12, 13}, {14, 15}, {16, 18}, {28, 36}});
  IntervalSet<int> set2(
      {{0, 1}, {4, 5}, {10, 12}, {15, 16}, {18, 28}, {40, 45}});

  EXPECT_EQ(set1.BoundingInterval(), Interval(2, 36));
  EXPECT_EQ(set2.BoundingInterval(), Interval(0, 45));

  EXPECT_TRUE(set1.Merge(set2));
  EXPECT_THAT(
      set1, ElementsAre(Interval(0, 1), Interval(2, 5), Interval(6, 8),
                        Interval(10, 13), Interval(14, 36), Interval(40, 45)));
  EXPECT_EQ(set1.BoundingInterval(), Interval(0, 45));

  IntervalSet<int> set3({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  IntervalSet<int> set4({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  EXPECT_TRUE(set3.Merge(set4));
  EXPECT_THAT(set3, ElementsAre(Interval(0, 8)));
  EXPECT_EQ(set3.BoundingInterval(), Interval(0, 8));
}

TEST(IntervalSetTest, MergeWithEmptySet) {
  IntervalSet<int> set1;
  IntervalSet<int> set2;
  EXPECT_TRUE(set1.Merge(set2));
  EXPECT_THAT(set1, ElementsAre());
  EXPECT_TRUE(set1.BoundingInterval().empty());

  IntervalSet<int> set3({{1, 2}});
  EXPECT_TRUE(set1.Merge(set3));
  EXPECT_THAT(set1, ElementsAre(Interval(1, 2)));
  EXPECT_EQ(set1.BoundingInterval(), Interval(1, 2));

  IntervalSet<int> set4;
  EXPECT_TRUE(set1.Merge(set4));
  EXPECT_THAT(set1, ElementsAre(Interval(1, 2)));
  EXPECT_EQ(set1.BoundingInterval(), Interval(1, 2));
}

TEST(IntervalSetTest, MergeOverlapping) {
  IntervalSet<int> set1({{2, 4}});
  IntervalSet<int> set2({{3, 5}});
  EXPECT_FALSE(set1.Merge(set2));
  EXPECT_THAT(set1, ElementsAre(Interval(2, 4)));
  EXPECT_EQ(set1.BoundingInterval(), Interval(2, 4));
}

TEST(IntervalSetTest, Clear) {
  IntervalSet<int> set({{1, 5}, {6, 9}});
  EXPECT_EQ(set.BoundingInterval(), Interval(1, 9));
  set.Clear();
  EXPECT_TRUE(set.empty());
  EXPECT_TRUE(set.BoundingInterval().empty());
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
