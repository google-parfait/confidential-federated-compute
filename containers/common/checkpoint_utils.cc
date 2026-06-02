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

#include "containers/common/checkpoint_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute {

using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;

absl::StatusOr<std::string> GetPrivacyId(CheckpointParser& parser) {
  FCP_ASSIGN_OR_RETURN(Tensor privacy_id_tensor,
                       parser.GetTensor(kPrivacyIdColumnName));
  if (privacy_id_tensor.dtype() != DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kPrivacyIdColumnName));
  }
  if (!privacy_id_tensor.is_scalar()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a scalar", kPrivacyIdColumnName));
  }
  return std::string(privacy_id_tensor.AsScalar<absl::string_view>());
}

absl::StatusOr<std::vector<std::string>> GetEventTime(
    CheckpointParser& parser, absl::string_view on_device_query_name) {
  // All checkpoints, including message-based ones, represent the event time as
  // a 1D string Tensor.
  const std::string time_tensor_name =
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName);
  FCP_ASSIGN_OR_RETURN(Tensor time_tensor, parser.GetTensor(time_tensor_name));
  if (time_tensor.dtype() != DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", time_tensor_name));
  }
  if (time_tensor.shape().dim_sizes().size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must have one dimension", time_tensor_name));
  }
  return time_tensor.ToStringVector();
}

}  // namespace confidential_federated_compute
