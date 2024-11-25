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

#include "absl/status/status.h"
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

using ::confidential_federated_compute::sql::TensorColumn;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BYTES;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;

// Replace the values of the column with keyed hashes.
absl::Status HashColumn(TensorColumn& column, absl::string_view key) {
  std::unique_ptr<MutableStringData> hashed_column =
      std::make_unique<MutableStringData>(column.tensor_.num_elements());
  absl::Span<const absl::string_view> column_span =
      column.tensor_.AsSpan<absl::string_view>();

  for (const absl::string_view value : column_span) {
    FCP_ASSIGN_OR_RETURN(std::string hashed_value, KeyedHash(value, key));
    hashed_column->Add(std::move(hashed_value));
  }

  FCP_ASSIGN_OR_RETURN(
      Tensor hashed_tensor,
      Tensor::Create(column.tensor_.dtype(), {column.tensor_.num_elements()},
                     std::move(hashed_column)));
  column.tensor_ = std::move(hashed_tensor);
  return absl::OkStatus();
}

}  // namespace

absl::Status HashSensitiveColumns(std::vector<TensorColumn>& contents,
                                  absl::string_view key) {
  for (TensorColumn& column : contents) {
    if (absl::StartsWith(column.column_schema_.name(), "SENSITIVE_")) {
      if (column.column_schema_.type() !=
              ExampleQuerySpec_OutputVectorSpec_DataType_STRING &&
          column.column_schema_.type() !=
              ExampleQuerySpec_OutputVectorSpec_DataType_BYTES) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Only STRING or BYTES types are supported for sensitive columns. "
            "Column %s has type %d",
            column.column_schema_.name(), column.column_schema_.type()));
      }
      FCP_RETURN_IF_ERROR(HashColumn(column, key));
    }
  }
  return absl::OkStatus();
}
}  // namespace confidential_federated_compute::fed_sql
