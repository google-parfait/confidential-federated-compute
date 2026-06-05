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
#include <optional>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/interval_set.h"

namespace confidential_federated_compute::fed_sql {

// An ordered map from non-overlapping intervals to values.
//
// Maintains the invariant that adjacent intervals with equal values are always
// merged into a single interval. This means the representation is always
// canonical: there is exactly one way to represent any given mapping.
//
// The key type T must be arithmetic. The value type V must support:
//   - operator== (for merge checks)
//
// Example usage:
//   IntervalMap<uint64_t, int> map;
//   // Insert intervals with values.
//   map.Insert({100, 200}, 5);
//   map.Insert({200, 300}, 3);
//   // Decrement stored values in [100, 200).
//   map.ForEachValue({100, 200}, [](int& v) { v--; return true; });
//   // Check whether all stored values in [50, 250) are positive.
//   bool ok = map.ForEachValue({50, 250}, [](int& v) { return v > 0; });
template <typename T, typename V, typename>
class IntervalMap {
 public:
  using InnerMap = absl::btree_map<Interval<T>, V, IntervalLess<T>>;
  using const_iterator = typename InnerMap::const_iterator;
  using value_type = typename InnerMap::value_type;

  // Creates an empty IntervalMap.
  IntervalMap() = default;

  // Creates an IntervalMap from a list of non-overlapping and ordered
  // (interval, value) pairs. Adjacent intervals with the same value will be
  // merged. Empty intervals are ignored.
  explicit IntervalMap(std::initializer_list<std::pair<Interval<T>, V>> il) {
    for (const auto& [interval, value] : il) {
      if (interval.empty()) continue;

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

  // Equality operators.
  bool operator==(const IntervalMap& other) const { return map_ == other.map_; }
  bool operator!=(const IntervalMap& other) const { return !(*this == other); }

  // IntervalMap's begin() iterator.
  const_iterator begin() const { return map_.begin(); }

  // IntervalMap's end() iterator.
  const_iterator end() const { return map_.end(); }

  // Returns the number of disjoint intervals explicitly stored in this
  // IntervalMap.
  size_t size() const { return map_.size(); }

  bool empty() const { return map_.empty(); }

  void Clear() { map_.clear(); }

  // Returns the end of the last interval or std::nullopt if empty.
  std::optional<T> last_interval_end() const {
    if (map_.empty()) return std::nullopt;
    return map_.rbegin()->first.end();
  }

  // Applies `func` to every stored value within the given interval.
  // Intervals that are only partially covered by the given interval are split
  // at the interval boundaries. After mutation, adjacent intervals with equal
  // values are merged.
  //
  // `func` must be callable as `bool(V&)`. It receives a mutable reference
  // to each stored value and returns true to continue, or false to abort. If
  // `func` returns false, the map may be left in a partially-mutated state.
  //
  // Returns true if `func` returned true for every stored sub-interval, or if
  // no stored intervals overlap the given interval. Returns false as soon as
  // `func` returns false.
  //
  // Note: gaps (sub-intervals not covered by any stored entry) are NOT visited.
  //
  // Example — decrement every stored value by 1:
  //   map.ForEachValue(interval, [](V& val) { val--; return true; });
  //
  // Example — check whether all stored values are positive (read-only):
  //   map.ForEachValue(interval, [](V& val) { return val > 0; });
  template <typename Func>
  bool ForEachValue(Interval<T> interval, Func func) {
    // Empty intervals are a no-op.
    if (interval.empty()) return true;

    // Split any intervals that straddle the interval boundaries.
    CutAt(interval.start());
    CutAt(interval.end());

    // Apply func to all stored intervals within [start, end).
    auto it = map_.lower_bound(interval.start());
    while (it != map_.end() && it->first.start() < interval.end()) {
      if (!func(it->second)) {
        // Merge adjacent intervals with equal values.
        CleanupAfterMutation(interval);
        return false;
      }
      ++it;
    }

    // Merge adjacent intervals with equal values.
    CleanupAfterMutation(interval);
    return true;
  }

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

  // Inserts `interval` with `value`. Adjacent intervals with equal values are
  // merged at the boundaries. Empty intervals are a no-op (returns true).
  //
  // Returns false without modifying the map if `interval` overlaps any
  // existing stored interval.
  bool Insert(Interval<T> interval, V value) {
    if (interval.empty()) return true;

    // Check for overlap before touching the map.
    auto it = map_.lower_bound(interval.start());

    // A stored interval that starts before interval.start() might still extend
    // into the new interval.
    if (it != map_.begin()) {
      auto prev = std::prev(it);
      if (prev->first.end() > interval.start()) return false;
    }

    // Any stored interval that starts strictly before interval.end() overlaps.
    if (it != map_.end() && it->first.start() < interval.end()) return false;

    map_.emplace(interval, value);
    TryMergeAt(interval.start());
    TryMergeAt(interval.end());
    return true;
  }

  // Erases all intervals for which the predicate returns true.
  template <typename Predicate>
  void EraseIf(Predicate pred) {
    std::vector<Interval<T>> to_erase;
    for (auto it = map_.begin(); it != map_.end(); ++it) {
      if (pred(*it)) {
        to_erase.push_back(it->first);
      }
    }

    for (const auto& key : to_erase) {
      map_.erase(key);
    }
  }

 private:
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

  // Merges adjacent intervals with equal values.
  void CleanupAfterMutation(Interval<T> interval) {
    // Merge adjacent intervals with equal values within the interval.
    // Note: absl::btree_map invalidates all iterators on mutation, so we
    // must re-acquire iterators after any erase.
    auto it = map_.lower_bound(interval.start());
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

  InnerMap map_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_MAP_H_
