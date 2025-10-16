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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_SET_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_SET_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "containers/sql/input.h"
#include "containers/sql/row_view.h"
#include "fcp/base/monitoring.h"

namespace confidential_federated_compute::sql {
struct RowLocation {
  uint64_t dp_unit_hash;  // Hash of the DP unit identifiers.
  uint32_t input_index;   // Index into the primary Input storage
                          // vector.
  uint32_t row_index;     // Index of the row within the tensors of the
                          // Input.

  // Sort by DP unit hash, then input index, then row index. First sorting by DP
  // unit hash lets us group rows by DP unit so we can process them together.
  // Secondarily grouping by input index and row index helps with memory
  // locality as we iterate over rows.
  bool operator<(const RowLocation& other) const {
    return std::tie(dp_unit_hash, input_index, row_index) <
           std::tie(other.dp_unit_hash, other.input_index, other.row_index);
  }
};

// A non-owning view of a set of rows from a collection of inputs. All Inputs
// must hold the same set of column names in the same order.
class RowSet {
 public:
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = RowView;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    // The result of operator* is a temporary, so we can't return a real
    // reference.
    using reference = value_type;

    static Iterator Begin(absl::Span<const RowLocation> locations,
                          absl::Span<const Input> storage) {
      return Iterator(locations.data(), storage);
    }

    static Iterator End(absl::Span<const RowLocation> locations,
                        absl::Span<const Input> storage) {
      return Iterator(locations.data() + locations.size(), storage);
    }

    reference operator*() const {
      const Input& input = all_inputs_[current_location_->input_index];
      absl::StatusOr<RowView> row = input.GetRow(current_location_->row_index);
      FCP_CHECK(row.ok()) << row.status();
      return *row;
    }

    Iterator& operator++() {
      ++current_location_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator old = *this;
      ++(*this);
      return old;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.current_location_ == b.current_location_;
    }

    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return !(a == b);
    }

   private:
    Iterator(const RowLocation* current_location,
             absl::Span<const Input> storage)
        : current_location_(current_location), all_inputs_(storage) {}
    const RowLocation* current_location_;
    absl::Span<const Input> all_inputs_;
  };

  static absl::StatusOr<RowSet> Create(absl::Span<const RowLocation> locations,
                                       absl::Span<const Input> storage);

  Iterator begin() const { return Iterator::Begin(locations_, storage_); }
  Iterator end() const { return Iterator::End(locations_, storage_); }

  size_t size() const { return locations_.size(); }

  RowSet subspan(size_t pos = 0,
                 size_t count = absl::Span<const RowLocation>::npos) const {
    return RowSet(locations_.subspan(pos, count), storage_);
  }

  absl::StatusOr<absl::Span<const std::string>> GetColumnNames() const;

 private:
  RowSet(absl::Span<const RowLocation> locations,
         absl::Span<const Input> storage)
      : locations_(locations), storage_(storage) {}

  absl::Span<const RowLocation> locations_;
  absl::Span<const Input> storage_;
};

}  // namespace confidential_federated_compute::sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_SET_H_
