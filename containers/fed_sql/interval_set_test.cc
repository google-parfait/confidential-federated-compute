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

#include "containers/fed_sql/interval_set.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::testing::ElementsAre;

TEST(IntervalSetTest, EmptySet) {
  IntervalSet<int> set;
  EXPECT_EQ(set.size(), 0);
  EXPECT_THAT(set, ElementsAre());
  EXPECT_FALSE(set.Contains(0));
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
}

TEST(IntervalSetTest, EmptyIntervalsIgnored) {
  IntervalSet<int> set;
  set.Add(Interval<int>());
  set.Add({10, 10});
  EXPECT_THAT(set, ElementsAre());
}

TEST(IntervalSetTest, AddMultiple) {
  IntervalSet<int> set;
  set.Add({3, 4});
  set.Add({5, 6});
  set.Add({1, 2});
  set.Add({10, 11});
  EXPECT_THAT(set, ElementsAre(Interval(1, 2), Interval(3, 4), Interval(5, 6),
                               Interval(10, 11)));

  set.Add({11, 12});
  EXPECT_THAT(set, ElementsAre(Interval(1, 2), Interval(3, 4), Interval(5, 6),
                               Interval(10, 12)));

  set.Add({4, 5});
  EXPECT_THAT(set,
              ElementsAre(Interval(1, 2), Interval(3, 6), Interval(10, 12)));

  set.Add({7, 8});
  EXPECT_THAT(set, ElementsAre(Interval(1, 2), Interval(3, 6), Interval(7, 8),
                               Interval(10, 12)));

  set.Add({5, 9});
  EXPECT_THAT(set,
              ElementsAre(Interval(1, 2), Interval(3, 9), Interval(10, 12)));

  set.Add({1, 15});
  EXPECT_THAT(set, ElementsAre(Interval(1, 15)));

  set.Add({0, 1});
  EXPECT_THAT(set, ElementsAre(Interval(0, 15)));

  set.Add({20, 22});
  EXPECT_THAT(set, ElementsAre(Interval(0, 15), Interval(20, 22)));

  set.Add({-2, 10});
  EXPECT_THAT(set, ElementsAre(Interval(-2, 15), Interval(20, 22)));

  set.Add({17, 21});
  EXPECT_THAT(set, ElementsAre(Interval(-2, 15), Interval(17, 22)));

  set.Add({10, 20});
  EXPECT_THAT(set, ElementsAre(Interval(-2, 22)));
}

TEST(IntervalSetTest, Merge) {
  IntervalSet<int> set1(
      {{2, 4}, {6, 8}, {12, 13}, {14, 15}, {16, 18}, {28, 36}});
  IntervalSet<int> set2({{0, 1}, {3, 5}, {7, 9}, {10, 22}, {24, 26}, {30, 35}});

  set1.Merge(set2);
  EXPECT_THAT(
      set1, ElementsAre(Interval(0, 1), Interval(2, 5), Interval(6, 9),
                        Interval(10, 22), Interval(24, 26), Interval(28, 36)));

  IntervalSet<int> set3({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  IntervalSet<int> set4({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  set3.Merge(set4);
  EXPECT_THAT(set3, ElementsAre(Interval(0, 8)));
}

TEST(IntervalSetTest, MergeWithEmptySet) {
  IntervalSet<int> set1;
  IntervalSet<int> set2;
  set1.Merge(set2);
  EXPECT_THAT(set1, ElementsAre());

  IntervalSet<int> set3({{1, 2}});
  set1.Merge(set3);
  EXPECT_THAT(set1, ElementsAre(Interval(1, 2)));

  IntervalSet<int> set4;
  set1.Merge(set4);
  EXPECT_THAT(set1, ElementsAre(Interval(1, 2)));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
