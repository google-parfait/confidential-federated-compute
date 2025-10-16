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

#include "containers/sql/row_set.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "containers/sql/input.h"

namespace confidential_federated_compute::sql {

absl::StatusOr<RowSet> RowSet::Create(absl::Span<const RowLocation> locations,
                                      absl::Span<const Input> storage) {
  if (storage.size() > 1) {
    const auto first_columns = storage[0].GetColumnNames();
    for (size_t i = 1; i < storage.size(); ++i) {
      if (storage[i].GetColumnNames() != first_columns) {
        return absl::InvalidArgumentError(
            "All Inputs to a RowSet must have the same columns in the same "
            "order.");
      }
    }
  }
  return RowSet(locations, storage);
}

absl::StatusOr<absl::Span<const std::string>> RowSet::GetColumnNames() const {
  if (storage_.empty()) {
    return absl::Span<const std::string>();
  }
  return storage_[0].GetColumnNames();
}

}  // namespace confidential_federated_compute::sql
