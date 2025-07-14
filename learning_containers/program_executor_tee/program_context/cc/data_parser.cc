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

#include "program_executor_tee/program_context/cc/data_parser.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/crypto.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

using ::confidential_federated_compute::BlobDecryptor;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

absl::StatusOr<tensorflow_federated::v0::Value>
DataParser::ParseReadResponseToValue(
    const fcp::confidentialcompute::outgoing::ReadResponse& read_response,
    const std::string& nonce, const std::string& key) {
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(std::string fc_checkpoint,
                       ParseReadResponseToFcCheckpoint(read_response, nonce));
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<CheckpointParser> parser,
      parser_factory.Create(absl::Cord(std::move(fc_checkpoint))));
  FCP_ASSIGN_OR_RETURN(tensorflow_federated::aggregation::Tensor agg_tensor,
                       parser->GetTensor(key));
  return ConvertAggCoreTensorToValue(std::move(agg_tensor));
}

absl::StatusOr<tensorflow_federated::v0::Value>
DataParser::ConvertAggCoreTensorToValue(const Tensor& tensor) {
  tensorflow_federated::v0::Value value;

  federated_language::Array* array = value.mutable_array();

  federated_language::ArrayShape* array_shape = array->mutable_shape();
  for (int dim_size : tensor.shape().dim_sizes()) {
    array_shape->mutable_dim()->Add(dim_size);
  }
  array_shape->set_unknown_rank(false);

  switch (tensor.dtype()) {
    case tensorflow_federated::aggregation::DT_FLOAT: {
      array->set_dtype(federated_language::DataType::DT_FLOAT);
      auto float_values = tensor.AsSpan<float>();
      array->mutable_float32_list()->mutable_value()->Assign(
          float_values.begin(), float_values.end());
      break;
    }
    case tensorflow_federated::aggregation::DT_DOUBLE: {
      array->set_dtype(federated_language::DataType::DT_DOUBLE);
      auto double_values = tensor.AsSpan<double>();
      array->mutable_float64_list()->mutable_value()->Assign(
          double_values.begin(), double_values.end());
      break;
    }
    case tensorflow_federated::aggregation::DT_INT32: {
      array->set_dtype(federated_language::DataType::DT_INT32);
      auto int32_values = tensor.AsSpan<int32_t>();
      array->mutable_int32_list()->mutable_value()->Assign(int32_values.begin(),
                                                           int32_values.end());
      break;
    }
    case tensorflow_federated::aggregation::DT_INT64: {
      array->set_dtype(federated_language::DataType::DT_INT64);
      auto int64_values = tensor.AsSpan<int64_t>();
      array->mutable_int64_list()->mutable_value()->Assign(int64_values.begin(),
                                                           int64_values.end());
      break;
    }
    case tensorflow_federated::aggregation::DT_STRING: {
      array->set_dtype(federated_language::DataType::DT_STRING);
      auto string_values = tensor.AsSpan<absl::string_view>();
      array->mutable_string_list()->mutable_value()->Assign(
          string_values.begin(), string_values.end());
      break;
    }
    case tensorflow_federated::aggregation::DT_UINT64: {
      array->set_dtype(federated_language::DataType::DT_UINT64);
      auto uint64_values = tensor.AsSpan<uint64_t>();
      array->mutable_uint64_list()->mutable_value()->Assign(
          uint64_values.begin(), uint64_values.end());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found: ", tensor.dtype()));
  }

  return value;
}

absl::StatusOr<std::string> DataParser::ParseReadResponseToFcCheckpoint(
    const ReadResponse& read_response, const std::string& nonce) {
  if (read_response.first_response_metadata().has_unencrypted()) {
    return read_response.data();
  }
  if (read_response.first_response_metadata()
          .hpke_plus_aead_data()
          .rewrapped_symmetric_key_associated_data()
          .nonce() != nonce) {
    return absl::InvalidArgumentError("ReadResponse nonce does not match");
  }
  return blob_decryptor_->DecryptBlob(read_response.first_response_metadata(),
                                      read_response.data());
}

}  // namespace confidential_federated_compute::program_executor_tee