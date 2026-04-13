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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_H_

#include <algorithm>
#include <limits>
#include <ostream>
#include <type_traits>

#include "absl/log/check.h"

namespace confidential_federated_compute::fed_sql {

// Forward declaration of IntervalSet.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class IntervalSet;

// Forward declaration of IntervalMap.
template <typename T, typename V,
          typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class IntervalMap;

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
  template <typename, typename, typename>
  friend class IntervalMap;
  // Used by IntervalSet and IntervalMap to extend the end of the interval.
  // Despite the Interval being used as a key in a btree set/map, it's safe to
  // mutate the end since these containers only order by the start of each
  // interval.
  void extend_end(T end) const { end_ = std::max(end_, end); }
  // Used by IntervalSet and IntervalMap to shorten the end of the interval.
  void shorten_end(T end) const { end_ = std::min(end_, end); }

  T start_;
  mutable T end_;
};

template <typename T>
auto operator<<(std::ostream& out, const Interval<T>& i)
    -> decltype(out << i.start()) {
  return out << "[" << i.start() << ", " << i.end() << ")";
}

// Comparator for Interval<T> with heterogeneous lookup
// (https://abseil.io/tips/144). Since intervals are non-overlapping in the
// containers, we can compare intervals based on their start.
//
// Shared by IntervalSet and IntervalMap.
template <typename T>
struct IntervalLess {
  using is_transparent = void;

  bool operator()(const Interval<T>& a, const Interval<T>& b) const {
    return a.start() < b.start();
  }

  // Transparent overload for an implicit point interval `Interval<T>(a, a)`.
  bool operator()(const T& a, const Interval<T>& b) const {
    return a < b.start();
  }

  // Transparent overload for an implicit point interval `Interval<T>(b, b)`.
  bool operator()(const Interval<T>& a, const T& b) const {
    return a.start() < b;
  }
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INTERVAL_H_
