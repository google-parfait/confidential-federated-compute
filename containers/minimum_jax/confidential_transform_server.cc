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
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::minimum_jax {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
SimpleSession::SessionWrite(
    const fcp::confidentialcompute::WriteRequest& write_request,
    std::string unencrypted_data) {
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
  if (t->num_elements() != 1) {
    return absl::InvalidArgumentError("Tensor contains more than 1 elements.");
  }
  absl::string_view s = t->CastToScalar<absl::string_view>();
  data_.push_back(std::string(s));

  return ToSessionWriteFinishedResponse(
      absl::OkStatus(),
      write_request.first_request_metadata().total_size_bytes());
}

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
SimpleSession::FinalizeSession(
    const fcp::confidentialcompute::FinalizeRequest& request,
    const fcp::confidentialcompute::BlobMetadata& input_metadata) {
  pybind11::scoped_interpreter guard{};

  try {
    pybind11::module_ tokens_lib = pybind11::module_::import("tokens");

    pybind11::object result_obj =
        tokens_lib.attr("find_most_frequent_token")(data_);

    std::string most_frequent_token = result_obj.cast<std::string>();
    LOG(INFO) << "The most frequent token is " << most_frequent_token
              << std::endl;

    SessionResponse session_response;
    ReadResponse* response = session_response.mutable_read();
    response->set_finish_read(true);
    *(response->mutable_data()) = most_frequent_token;
    return session_response;
  } catch (pybind11::error_already_set& e) {
    LOG(ERROR) << "Python error: " << e.what() << std::endl;
    return absl::InvalidArgumentError(e.what());
  }
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