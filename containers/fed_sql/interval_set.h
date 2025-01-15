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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_SET_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_SET_H_

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"

namespace confidential_federated_compute::fed_sql {

// Forward declaration of IntervalSet.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class IntervalSet;

// A simple numeric type interval with exclusive end.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class Interval {
 public:
  Interval() : start_(), end_() {}
  Interval(std::tuple<T, T> interval)
      : Interval(std::get<0>(interval), std::get<1>(interval)) {}
  Interval(T start, T end) : start_(start), end_(end) {
    CHECK_LE(start, end) << "Interval must have start <= end";
  }

  // The start is inclusive.
  T start() const { return start_; };
  // The end is exclusive.
  T end() const { return end_; };

  // Returns true if the interval is empty.
  bool empty() const { return start_ == end_; }

  friend bool operator==(const Interval &a, const Interval &b) {
    return a.start_ == b.start_ && a.end_ == b.end_;
  }

 private:
  friend class IntervalSet<T>;
  // Used by IntervalSet to extend the end of the interval.
  void extend_end(T end) { end_ = std::max(end_, end); }

  T start_;
  T end_;
};

// Ordered set of non-overlapping intervals.
// This class implements union of overlapping or contiguous intervals into
// a single interval.
//   For example: [1, 2) and [2, 3) -> [1, 3)
//                [1, 5) and [3, 7) -> [1, 7)
//                [1, 9) and [3, 5) -> [1, 9)
template <typename T, typename>
class IntervalSet {
 private:
  // Comparer with heterogeneous lookup (https://abseil.io/tips/144).
  // Since intervals are non-overlapping in the map, we can compare
  // intervals based on their start.
  struct IntervalLess {
    using is_transparent = void;
    // Implementation of the "less" operator
    bool operator()(const Interval<T> &a, const Interval<T> &b) const {
      return a.start() < b.start();
    }

    // Transparent overload  for an implicit point
    // interval `Interval<T>(a, a)`.
    bool operator()(const T &a, const Interval<T> &b) const {
      return a < b.start();
    }

    // Transparent overload for an implicit point interval `Interval<T>(b, b)`.
    bool operator()(const Interval<T> &a, const T &b) const {
      return a.start() < b;
    }
  };

 public:
  // Intervals are stored in a btree set, ordered by their start.
  using InnerSet = absl::btree_set<Interval<T>, IntervalLess>;
  using const_iterator = typename InnerSet::const_iterator;
  using value_type = Interval<T>;

  IntervalSet() = default;
  IntervalSet(std::initializer_list<Interval<T>> il) {
    Assign(il.begin(), il.end());
  }

  // IntervalSet's begin() iterator.
  const_iterator begin() const { return set_.begin(); }

  // IntervalSet's end() iterator.
  const_iterator end() const { return set_.end(); }

  // Returns the number of disjoint intervals contained in this IntervalSet.
  size_t size() const { return set_.size(); }

  void Clear() { set_.clear(); }

  template <typename Iter>
  void Assign(Iter first, Iter last) {
    Clear();
    for (; first != last; ++first) Add(*first);
  }

  // Checks if the value is contained in the set.
  bool Contains(T v) const {
    // The first interval that starts after the end of the inserted interval.
    auto it_next = set_.upper_bound(v);
    if (it_next == set_.begin()) {
      // The value is before the first interval.
      return false;
    }
    // The set contains the value if the previous interval end is after the
    // specified value.
    auto it_prev = std::prev(it_next);
    return it_prev->end() > v;
  }

  // Adds a new interval to the set. If the new interval overlaps or contiguous
  // with any of existing intervals, it gets merged.
  void Add(Interval<T> interval) {
    // This method implements the following cases:
    // 1) Overlap with the interval before the start of inserted interval.
    // 2) Overlap with the interval before the end of the inserted interval.
    // 3) Inserted interval is Within a single existing interval.
    // 4) No overlap between the inserted interval and any existing interval.

    // In cases (1) or (2) there may be one or more intervals that need to
    // be merged.
    // In general, we need to do the following:
    // * If the preceding interval overlaps with `interval`, extend the
    //   preceding interval to cover all overlapping intervals.
    // * If the preceding interval does not overlap with `interval`,
    //   extend `interval` to cover any overlapping intervals and insert
    //  `interval`
    // * Delete all other intervals that overlap with `interval

    // Empty intervals are ignored.
    if (interval.empty()) {
      return;
    }

    // The first interval that starts after the end of the inserted interval.
    auto it_next = set_.upper_bound(interval.end());
    if (it_next == set_.begin()) {
      // The inserted interval is before the first interval in the set.
      set_.insert(interval);
      return;
    }

    // The interval that starts before the end of the inserted interval and
    // potentially overlaps it.
    auto it_end = std::prev(it_next);
    interval.extend_end(it_end->end());

    // The interval that starts after the start of the inserted interval.
    auto it_start = set_.upper_bound(interval.start());
    bool insert = false;
    if (it_start == set_.begin()) {
      // There is no preceding interval.
      insert = true;
    } else {
      // Preceding interval that starts before the start of the inserted
      // interval.
      auto it_prev = std::prev(it_start);
      if (it_prev->end() < interval.start()) {
        // The inserted interval does not overlap with the preceding interval.
        insert = true;
      } else {
        // The inserted interval overlaps with the preceding interval.
        // Merge the two intervals.
        it_prev->extend_end(interval.end());
      }
    }
    // Delete all intervals after the start of the inserted or merged interval
    // and up to but not including the end of the inserted or merged interval.
    set_.erase(it_start, it_next);
    if (insert) {
      // If necessary, insert the new interval.
      set_.insert(interval);
    }
  }

  // Merges this IntervalSet with another interval set.
  void Merge(const IntervalSet<T> &other) {
    InnerSet this_set;
    std::swap(this_set, set_);
    const_iterator this_it = this_set.begin();
    const_iterator this_end = this_set.end();
    const_iterator other_it = other.set_.begin();
    const_iterator other_end = other.set_.end();

    // Implement the merge join of the two sets.
    std::optional<Interval<T>> cur, next;
    while (next = NextOfTwoSets(this_it, this_end, other_it, other_end),
           next.has_value()) {
      if (cur.has_value()) {
        if (next->start() <= cur->end()) {
          cur->extend_end(next->end());
          continue;
        } else {
          set_.insert(set_.end(), *std::move(cur));
        }
      }
      std::swap(cur, next);
    }

    if (cur.has_value()) {
      set_.insert(set_.end(), *std::move(cur));
    }
  }

 private:
  // Retrieves a next interval from two iterators based on which one
  // has a smaller start of the interval.  This is used for merging two sets.
  static std::optional<Interval<T>> NextOfTwoSets(const_iterator &it1,
                                                  const_iterator end1,
                                                  const_iterator &it2,
                                                  const_iterator end2) {
    if (it1 != end1) {
      if (it2 != end2 && it2->start() < it1->start()) {
        return *it2++;
      } else {
        return *it1++;
      }
    } else if (it2 != end2) {
      return *it2++;
    }
    // Reached the end of both sets.
    return std::nullopt;
  }
  // Internal set of intervals.
  InnerSet set_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_SET_H_
