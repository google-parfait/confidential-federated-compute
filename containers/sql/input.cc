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

#include "containers/sql/input.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "containers/sql/row_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

Input::Input(ContentsVariant contents,
             fcp::confidentialcompute::BlobHeader blob_header,
             std::vector<std::string> column_names)
    : contents_(std::move(contents)),
      blob_header_(std::move(blob_header)),
      column_names_(std::move(column_names)) {}

absl::StatusOr<Input> Input::CreateFromTensors(
    std::vector<tensorflow_federated::aggregation::Tensor> contents,
    fcp::confidentialcompute::BlobHeader blob_header) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("No columns provided.");
  }
  if (contents[0].shape().dim_sizes().empty()) {
    return absl::InvalidArgumentError("Column has no rows.");
  }

  size_t num_rows = contents[0].shape().dim_sizes()[0];
  std::vector<std::string> column_names;
  for (const auto& column : contents) {
    column_names.push_back(column.name());
    if (column.shape().dim_sizes().empty()) {
      return absl::InvalidArgumentError("Column has no rows.");
    }
    if (column.shape().dim_sizes().size() > 1) {
      return absl::InvalidArgumentError("Column has more than one dimension.");
    }
    if (column.shape().dim_sizes()[0] != num_rows) {
      return absl::InvalidArgumentError(
          "All columns must have the same number of rows.");
    }
  }
  return Input(TensorContents(std::move(contents)), std::move(blob_header),
               std::move(column_names));
}

absl::Span<const std::string> Input::GetColumnNames() const {
  return column_names_;
}

absl::StatusOr<RowView> Input::GetRow(uint32_t row_index) const {
  return absl::visit(
      [row_index](const auto& data) { return data.GetRow(row_index); },
      contents_);
}

size_t Input::GetRowCount() const {
  return absl::visit(
      [](const auto& data) -> size_t { return data.GetRowCount(); }, contents_);
}

std::vector<tensorflow_federated::aggregation::Tensor>
Input::MoveToTensors() && {
  return absl::visit(
      [](auto&& data)
          -> std::vector<tensorflow_federated::aggregation::Tensor> {
        return std::move(data).MoveToTensors();
      },
      std::move(contents_));
}

size_t Input::TensorContents::GetRowCount() const {
  if (contents_.empty()) {
    return 0;
  }
  return contents_[0].shape().dim_sizes()[0];
}

}  // namespace confidential_federated_compute::sql
