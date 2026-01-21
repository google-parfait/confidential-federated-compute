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
#include <limits>
#include <optional>
#include <ostream>
#include <type_traits>
#include <utility>  // For std::move

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

  friend bool operator==(const Interval& a, const Interval& b) {
    return a.start_ == b.start_ && a.end_ == b.end_;
  }

  // Returns true if the interval contains the specified value.
  bool Contains(T value) const {
    // There is a special case for interval end being at the max
    // value, in which case the value is considered to be included.
    return value >= start_ &&
           (value < end_ ||
            (end_ == std::numeric_limits<T>::max() && value == end_));
  }

 private:
  friend class IntervalSet<T>;
  // Used by IntervalSet to extend the end of the interval.
  // Despite the Interval being used as a key in a btree set, it's safe to
  // mutate the end since IntervalSet only orders by the start of each interval.
  void extend_end(T end) const { end_ = std::max(end_, end); }

  T start_;
  mutable T end_;
};

template <typename T>
auto operator<<(std::ostream& out, const Interval<T>& i)
    -> decltype(out << i.start()) {
  return out << "[" << i.start() << ", " << i.end() << ")";
}

// Ordered set of non-overlapping, non-adjacent intervals.
// This invariant is maintained when adding new intervals or joining with
// another set of intervals.
// Adding or merging overlapping intervals isn't allowed.
template <typename T, typename>
class IntervalSet {
 private:
  // Comparer with heterogeneous lookup (https://abseil.io/tips/144).
  // Since intervals are non-overlapping in the map, we can compare
  // intervals based on their start.
  struct IntervalLess {
    using is_transparent = void;
    // Implementation of the "less" operator
    bool operator()(const Interval<T>& a, const Interval<T>& b) const {
      return a.start() < b.start();
    }

    // Transparent overload  for an implicit point
    // interval `Interval<T>(a, a)`.
    bool operator()(const T& a, const Interval<T>& b) const {
      return a < b.start();
    }

    // Transparent overload for an implicit point interval `Interval<T>(b, b)`.
    bool operator()(const Interval<T>& a, const T& b) const {
      return a.start() < b;
    }
  };

 public:
  // Intervals are stored in a btree set, ordered by their start.
  using InnerSet = absl::btree_set<Interval<T>, IntervalLess>;
  using const_iterator = typename InnerSet::const_iterator;
  using value_type = Interval<T>;

  IntervalSet() = default;

  // Creates IntervalSet from a list of non-overlapping and ordered
  // intervals.
  IntervalSet(std::initializer_list<Interval<T>> il) {
    CHECK(Assign(il.begin(), il.end()));
  }

  // Equality operators.
  bool operator==(const IntervalSet& other) const { return set_ == other.set_; }
  bool operator!=(const IntervalSet& other) const { return !(*this == other); }

  // IntervalSet's begin() iterator.
  const_iterator begin() const { return set_.begin(); }

  // IntervalSet's end() iterator.
  const_iterator end() const { return set_.end(); }

  // Returns the number of disjoint intervals contained in this IntervalSet.
  size_t size() const { return set_.size(); }

  bool empty() const { return set_.empty(); }

  void Clear() { set_.clear(); }

  // Populates this IntervalSet with the specified intervals.
  // Intervals must be non-overlapping and ordered.
  template <typename Iter>
  bool Assign(Iter first, Iter last) {
    InnerSet set;
    std::optional<T> prev_end;

    for (; first != last; ++first) {
      if (prev_end.has_value() && first->start() <= *prev_end) {
        return false;
      }

      prev_end = first->end();
      set.insert(set.end(), *first);
    }
    std::swap(set_, set);
    return true;
  }

  // Checks if the value is contained in the set.
  bool Contains(T v) const {
    // The first interval that starts after the end of the inserted interval.
    auto it_next = set_.upper_bound(v);
    if (it_next == set_.begin()) {
      // The value is before the first interval.
      return false;
    }
    // The set contains the value if the previous interval contains the value.
    auto it_prev = std::prev(it_next);
    return it_prev->Contains(v);
  }

  // Adds a new interval to the set assuming that the interval doesn't already
  // exist or overlaps any existing interval. If the new interval overlaps with
  // any of existing intervals, this function returns false. If the new
  // interval is adjacent with any of existing intervals, it gets merged.
  bool Add(Interval<T> interval) {
    // Empty intervals are ignored.
    if (interval.empty()) {
      return true;
    }

    // The new interval is inserted in all cases except when it is adjacent
    // to the previous interval.
    bool insert_new = true;
    // The next interval (following the inserted interval, if any) is erased
    // when the new interval is adjacent to it.
    bool erase_next = false;

    // The first interval that starts after the start of the inserted interval.
    auto it_next = set_.upper_bound(interval.start());
    if (it_next != set_.end()) {
      if (it_next->start() < interval.end()) {
        // The inserted interval overlaps with the next interval.
        return false;
      } else if (it_next->start() == interval.end()) {
        // The inserted interval is adjacent to the next interval.
        interval.extend_end(it_next->end());
        erase_next = true;
      }
    }

    // Check if there is a previous interval
    if (it_next != set_.begin()) {
      auto it_prev = std::prev(it_next);
      if (it_prev->end() > interval.start()) {
        // The inserted interval overlaps with the previous interval.
        return false;
      } else if (it_prev->end() == interval.start()) {
        // The inserted interval is adjacent to the previous interval.
        it_prev->extend_end(interval.end());
        insert_new = false;
      }
    }

    if (erase_next) {
      set_.erase(it_next);
    }
    if (insert_new) {
      set_.insert(interval);
    }

    return true;
  }

  // Merges this IntervalSet with another interval set.
  // This returns false if there is any overlap between the two sets.
  bool Merge(const IntervalSet<T>& other) {
    InnerSet merged_set;
    const_iterator this_it = set_.begin();
    const_iterator this_end = set_.end();
    const_iterator other_it = other.set_.begin();
    const_iterator other_end = other.set_.end();

    // Implement the merge join of the two sets.
    std::optional<Interval<T>> cur, next;
    while (next = NextOfTwoSets(this_it, this_end, other_it, other_end),
           next.has_value()) {
      if (cur.has_value()) {
        if (next->start() < cur->end()) {
          // Overlap.
          return false;
        }
        if (next->start() == cur->end()) {
          cur->extend_end(next->end());
          continue;
        } else {
          merged_set.insert(merged_set.end(), *std::move(cur));
        }
      }
      std::swap(cur, next);
    }

    if (cur.has_value()) {
      merged_set.insert(merged_set.end(), *std::move(cur));
    }

    std::swap(merged_set, set_);
    return true;
  }

  // Returns the minimum bounding interval that covers all intervals in the set.
  // Returns an empty interval if the set is empty.
  Interval<T> BoundingInterval() const {
    if (set_.empty()) {
      return Interval<T>();  // Empty interval {0, 0}
    }
    // Because the set is ordered and intervals are non-overlapping,
    // the bounding interval starts with the first element's start
    // and ends with the last element's end.
    return Interval<T>(set_.begin()->start(), set_.rbegin()->end());
  }

 private:
  // Retrieves a next interval from two iterators based on which one
  // has a smaller start of the interval.  This is used for merging two sets.
  static std::optional<Interval<T>> NextOfTwoSets(const_iterator& it1,
                                                  const_iterator end1,
                                                  const_iterator& it2,
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
