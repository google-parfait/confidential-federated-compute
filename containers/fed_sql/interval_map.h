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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_MAP_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_MAP_H_

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/interval_set.h"

namespace confidential_federated_compute::fed_sql {

// An ordered map from non-overlapping intervals to values.
//
// Every point in the key space has a value: points not covered by any explicit
// interval have the default value provided at construction time.
//
// Maintains the invariant that adjacent intervals with equal values are always
// merged into a single interval, and intervals whose value equals the default
// are never stored. This means the representation is always canonical: there
// is exactly one way to represent any given mapping.
//
// The key type T must be arithmetic. The value type V must support:
//   - operator== (for merge checks)
//
// Example usage:
//   // Default value is 1 for all intervals.
//   IntervalMap<uint64_t, int> interval_map(/*default_value=*/1);
//   // Decrement [100, 200) by 1 using ForEachValue.
//   interval_map.ForEachValue({100, 200}, [](int& v) { v--; return true; });
//   // Check whether all values in [50, 150) are positive.
//   bool ok = interval_map.ForEachValue({50, 150},
//       [](int& v) { return v > 0; });
template <typename T, typename V, typename>
class IntervalMap {
 public:
  using InnerMap = absl::btree_map<Interval<T>, V, IntervalLess<T>>;
  using const_iterator = typename InnerMap::const_iterator;
  using value_type = typename InnerMap::value_type;

  // Creates an empty IntervalMap with a value-initialized default value.
  explicit IntervalMap(V default_value = V())
      : default_value_(std::move(default_value)) {}

  // Creates IntervalMap from a list of non-overlapping and ordered
  // (interval, value) pairs. Adjacent intervals with the same value
  // will be merged. Empty intervals and intervals with the default value
  // are ignored.
  IntervalMap(V default_value,
              std::initializer_list<std::pair<Interval<T>, V>> il)
      : default_value_(std::move(default_value)) {
    for (const auto& [interval, value] : il) {
      if (interval.empty()) continue;
      CHECK_LE(value, default_value_)
          << "Initial values must be less than or equal to default value";
      if (value == default_value_) continue;

      if (!map_.empty()) {
        auto last = std::prev(map_.end());
        CHECK_GE(interval.start(), last->first.end())
            << "Intervals must be non-overlapping and ordered";
        if (last->first.end() == interval.start() && last->second == value) {
          // Merge with the previous interval.
          last->first.extend_end(interval.end());
          continue;
        }
      }
      map_.emplace_hint(map_.end(), interval, value);
    }
  }

  // Returns the default value that the IntervalMap was initialized
  // with.
  const V& default_value() const { return default_value_; }

  // Equality operators.
  bool operator==(const IntervalMap& other) const {
    return default_value_ == other.default_value_ && map_ == other.map_;
  }
  bool operator!=(const IntervalMap& other) const { return !(*this == other); }

  // IntervalMap's begin() iterator.
  const_iterator begin() const { return map_.begin(); }

  // IntervalMap's end() iterator.
  const_iterator end() const { return map_.end(); }

  // Returns the number of disjoint intervals explicitly stored in this
  // IntervalMap (excludes the implicit default-valued intervals).
  size_t size() const { return map_.size(); }

  bool empty() const { return map_.empty(); }

  void Clear() { map_.clear(); }

  // Applies `func` to every value within the given interval, including
  // implicit default-valued gaps (which are materialized before mutation).
  // Intervals that are only partially covered by the given interval are split
  // at the interval boundaries. After mutation, adjacent intervals with equal
  // values are merged, and intervals equal to the default value are removed.
  //
  // `func` must be callable as `bool(V&)`. It receives a mutable reference
  // to each value and returns true to continue, or false to abort. If `func`
  // returns false, the map may be left in a partially-mutated state.
  //
  // Returns true if `func` returned true for every sub-interval, or if the
  // interval is empty. Returns false as soon as `func` returns false.
  //
  // Example — decrement every value by 1:
  //   map.ForEachValue(interval, [](V& val) { val--; return true; });
  //
  // Example — check whether all values are positive:
  //   bool ok = map.ForEachValue(interval, [](V& val) { return val > V{}; });
  template <typename Func>
  bool ForEachValue(Interval<T> interval, Func func) {
    // Empty intervals are a no-op.
    if (interval.empty()) return true;

    // Split any intervals that straddle the interval boundaries.
    CutAt(interval.start());
    CutAt(interval.end());

    // Fill gaps within the interval with the default value so that
    // func applies uniformly.
    for (const auto& gap : GetGaps(interval)) {
      map_.emplace(gap, default_value_);
    }

    // Apply func to all intervals within [start, end).
    auto it = map_.lower_bound(interval.start());
    while (it != map_.end() && it->first.start() < interval.end()) {
      if (!func(it->second)) {
        // Clean up: remove entries equal to default and merge boundaries.
        CleanupAfterMutation(interval);
        return false;
      }
      ++it;
    }

    // Clean up: remove entries equal to default and merge boundaries.
    CleanupAfterMutation(interval);
    return true;
  }

 private:
  // Returns the set of gaps (sub-intervals not covered by any stored interval)
  // within the given bounding interval.
  IntervalSet<T> GetGaps(Interval<T> interval) const {
    IntervalSet<T> gaps;
    T cursor = interval.start();
    auto it = map_.lower_bound(interval.start());

    // Check if the previous interval covers interval.start().
    if (it != map_.begin()) {
      auto prev = std::prev(it);
      if (prev->first.end() > cursor) {
        cursor = prev->first.end();
      }
    }

    while (cursor < interval.end()) {
      if (it == map_.end() || it->first.start() >= interval.end()) {
        gaps.Add({cursor, interval.end()});
        break;
      }
      if (it->first.start() > cursor) {
        gaps.Add({cursor, it->first.start()});
      }
      cursor = it->first.end();
      ++it;
    }
    return gaps;
  }

  // Splits any interval containing `point` into two intervals at `point`.
  // If `point` is already an interval boundary or is not contained in any
  // interval, this is a no-op.
  void CutAt(T point) {
    auto it = map_.upper_bound(point);
    if (it == map_.begin()) return;
    --it;

    const Interval<T>& interval = it->first;
    if (interval.start() == point) return;  // Already a boundary.
    if (interval.end() <= point) return;    // Doesn't contain point.

    // Split [start, end) -> v into [start, point) -> v and [point, end) -> v.
    V value = it->second;
    T end = interval.end();
    interval.shorten_end(point);
    map_.emplace(Interval<T>(point, end), value);
  }

  // Tries to merge two adjacent intervals meeting at `point`.
  // If the interval ending at `point` and the interval starting at `point`
  // exist and have equal values, they are merged into a single interval.
  void TryMergeAt(T point) {
    // Find the interval starting at `point`.
    auto it_right = map_.lower_bound(point);
    if (it_right == map_.begin() || it_right == map_.end()) return;
    if (it_right->first.start() != point) return;

    auto it_left = std::prev(it_right);
    if (it_left->first.end() != point) return;

    // Both intervals meet at `point`. Merge if values are equal.
    if (!(it_left->second == it_right->second)) return;

    // Extend the left interval to cover the right, then erase the right.
    it_left->first.extend_end(it_right->first.end());
    map_.erase(it_right);
  }

  // Removes entries that have been set back to the default value and
  // merges adjacent intervals with equal values.
  void CleanupAfterMutation(Interval<T> interval) {
    // Remove entries equal to the default value.
    auto it = map_.lower_bound(interval.start());
    while (it != map_.end() && it->first.start() < interval.end()) {
      if (it->second == default_value_) {
        it = map_.erase(it);
      } else {
        ++it;
      }
    }

    // Merge adjacent intervals with equal values within the interval.
    // Note: absl::btree_map invalidates all iterators on mutation, so we
    // must re-acquire iterators after any erase.
    it = map_.lower_bound(interval.start());
    while (it != map_.end() && it->first.start() < interval.end()) {
      auto next = std::next(it);
      if (next != map_.end() && next->first.start() < interval.end() &&
          it->first.end() == next->first.start() &&
          it->second == next->second) {
        T merged_end = next->first.end();
        T it_start = it->first.start();
        map_.erase(next);
        // Re-acquire `it` since erase invalidated all iterators.
        it = map_.lower_bound(it_start);
        it->first.extend_end(merged_end);
        // Don't advance — check if we can continue merging.
      } else {
        ++it;
      }
    }

    // Merge at the outer boundaries where values may now be equal.
    TryMergeAt(interval.end());
    TryMergeAt(interval.start());
  }

  V default_value_;
  InnerMap map_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_MAP_H_
