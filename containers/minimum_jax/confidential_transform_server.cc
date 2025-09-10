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
#include "confidential_transform_server.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/blob_metadata.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::minimum_jax {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

namespace {
// Similar to StringTensorDataAdapter but stores the original value in a vector.
class StringVectorTensorDataAdapter : public TensorData {
 public:
  explicit StringVectorTensorDataAdapter(const std::vector<std::string>* value)
      : string_views_(value->begin(), value->end()) {}

  size_t byte_size() const override {
    return string_views_.size() * sizeof(absl::string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  std::vector<absl::string_view> string_views_;
};

absl::StatusOr<Tensor> ConvertStringTensor(
    const std::vector<std::string>* value) {
  int64_t size = value->size();
  TensorShape tensor_shape({size});
  return Tensor::Create(tensorflow_federated::aggregation::DT_STRING,
                        tensor_shape,
                        std::make_unique<StringVectorTensorDataAdapter>(value));
}

}  // namespace

absl::StatusOr<std::tuple<BlobMetadata, std::string>> EncryptSessionResult(
    const BlobMetadata& input_metadata, absl::string_view unencrypted_result,
    uint32_t output_access_policy_node_id) {
  BlobEncryptor encryptor;
  BlobHeader input_header;
  if (!input_header.ParseFromString(
          input_metadata.hpke_plus_aead_data().ciphertext_associated_data())) {
    return absl::InvalidArgumentError(
        "Failed to parse the BlobHeader when trying to encrypt outputs.");
  }
  return encryptor.EncryptBlob(unencrypted_result,
                               input_metadata.hpke_plus_aead_data()
                                   .rewrapped_symmetric_key_associated_data()
                                   .reencryption_public_key(),
                               input_header.access_policy_sha256(),
                               output_access_policy_node_id);
}

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
SimpleSession::SessionWrite(
    const fcp::confidentialcompute::WriteRequest& write_request,
    std::string unencrypted_data) {
  FedSqlContainerWriteConfiguration write_config;
  if (!write_request.first_request_configuration().UnpackTo(&write_config)) {
    return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerWriteConfiguration."));
  }
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(std::move(unencrypted_data)));
  if (!parser.ok()) {
    return absl::InvalidArgumentError("Failed to parse the checkpoint");
  }
  auto t = (*parser)->GetTensor("data");
  if (!t.ok()) {
    return absl::InvalidArgumentError("Failed to create the tensor.");
  }

  absl::MutexLock mutexlock(mutex_);
  switch (write_config.type()) {
    case fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE: {
      if (t->num_elements() != 1) {
        return absl::InvalidArgumentError(
            "Tensor contains more than 1 elements.");
      }
      absl::string_view s = t->CastToScalar<absl::string_view>();
      data_.push_back(std::string(s));
      break;
    }

    case fcp::confidentialcompute::AGGREGATION_TYPE_MERGE: {
      auto new_data = t->ToStringVector();
      data_.insert(data_.end(), new_data.begin(), new_data.end());
      break;
    }

    default:
      return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
          "AggCoreAggregationType must be specified."));
  }

  return ToSessionWriteFinishedResponse(
      absl::OkStatus(),
      write_request.first_request_metadata().total_size_bytes());
}

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
SimpleSession::FinalizeSession(
    const fcp::confidentialcompute::FinalizeRequest& request,
    const fcp::confidentialcompute::BlobMetadata& input_metadata) {
  absl::MutexLock mutexlock(mutex_);
  FedSqlContainerFinalizeConfiguration finalize_config;
  if (!request.configuration().UnpackTo(&finalize_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerFinalizeConfiguration.");
  }

  std::string result;
  BlobMetadata result_metadata;

  switch (finalize_config.type()) {
    case fcp::confidentialcompute::FINALIZATION_TYPE_REPORT: {
      try {
        pybind11::scoped_interpreter guard{};
        pybind11::module_ tokens_lib = pybind11::module_::import("tokens");
        pybind11::object result_obj =
            tokens_lib.attr("find_most_frequent_token")(data_);
        auto unencrypted_result = result_obj.cast<std::string>();
        result = unencrypted_result;
        result_metadata.set_compression_type(
            BlobMetadata::COMPRESSION_TYPE_NONE);
        result_metadata.set_total_size_bytes(result.size());
        result_metadata.mutable_unencrypted();
        break;
      } catch (pybind11::error_already_set& e) {
        LOG(ERROR) << "Python error: " << e.what() << std::endl;
        return absl::InvalidArgumentError(e.what());
      }
    }

    case fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE: {
      FCP_ASSIGN_OR_RETURN(Tensor tensor, ConvertStringTensor(&data_));
      FederatedComputeCheckpointBuilderFactory factory;
      auto builder = factory.Create();
      FCP_RETURN_IF_ERROR(builder->Add("data", tensor));
      FCP_ASSIGN_OR_RETURN(absl::Cord ckpt, builder->Build());

      FCP_ASSIGN_OR_RETURN(
          std::tie(result_metadata, result),
          EncryptSessionResult(input_metadata, std::string(ckpt), 0));
      break;
    }

    default:
      return absl::InvalidArgumentError(
          "Finalize configuration must specify the finalization type.");
  }

  SessionResponse session_response;
  ReadResponse* response = session_response.mutable_read();
  response->set_finish_read(true);
  *(response->mutable_data()) = result;
  *(response->mutable_first_response_metadata()) = result_metadata;
  return session_response;
}

absl::StatusOr<std::string> SimpleConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  // Returns an empty string for unencrypted payloads.
  if (metadata.has_unencrypted()) {
    return "";
  }

  // GetKeyId is only supported for KMS-enabled transforms.
  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  fcp::confidentialcompute::BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  if (blob_header.key_id().empty()) {
    return absl::InvalidArgumentError(
        "Parsed BlobHeader has an empty 'key_id'");
  }

  return blob_header.key_id();
}

}  // namespace confidential_federated_compute::minimum_jax