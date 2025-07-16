// Copyright 2024 Google LLC.
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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "containers/crypto.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/status_converters.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;

// Replace the values of the column with keyed hashes.
absl::StatusOr<Tensor> HashColumn(Tensor& column, absl::string_view key) {
  std::unique_ptr<MutableStringData> hashed_column =
      std::make_unique<MutableStringData>(column.num_elements());
  absl::Span<const absl::string_view> column_span =
      column.AsSpan<absl::string_view>();

  for (const absl::string_view value : column_span) {
    FCP_ASSIGN_OR_RETURN(std::string hashed_value, KeyedHash(value, key));
    hashed_column->Add(std::move(hashed_value));
  }

  return Tensor::Create(column.dtype(),
                        {static_cast<int64_t>(column.num_elements())},
                        std::move(hashed_column), column.name());
}

}  // namespace

absl::Status HashSensitiveColumns(std::vector<Tensor>& contents,
                                  absl::string_view key) {
  for (Tensor& column : contents) {
    // Client upload columns are prefixed by <query_name>/ while server-side
    // data isn't.
    if (absl::StartsWith(column.name(), "SENSITIVE_") ||
        absl::StrContains(column.name(), "/SENSITIVE_")) {
      if (column.dtype() != DataType::DT_STRING) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Only DT_STRING types are supported for sensitive columns. "
            "Column %s has type %d",
            column.name(), column.dtype()));
      }
      FCP_ASSIGN_OR_RETURN(Tensor hashed_column, HashColumn(column, key));
      column = std::move(hashed_column);
    }
  }
  return absl::OkStatus();
}
}  // namespace confidential_federated_compute::fed_sql
