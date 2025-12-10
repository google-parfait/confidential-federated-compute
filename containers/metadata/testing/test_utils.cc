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
#include "containers/metadata/testing/test_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace confidential_federated_compute::metadata::testing {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidentialcompute::BlobMetadata;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;

Tensor BuildStringTensor(std::string name, std::vector<std::string> values) {
  auto data = std::make_unique<MutableStringData>(values.size());
  for (auto& value : values) {
    data->Add(std::move(value));
  }

  absl::StatusOr<Tensor> tensor =
      Tensor::Create(DataType::DT_STRING, {static_cast<int64_t>(values.size())},
                     std::move(data));
  CHECK_OK(tensor.status());
  CHECK_OK(tensor->set_name(std::move(name)));
  return *std::move(tensor);
}

Tensor BuildIntTensor(std::string name, std::initializer_list<int64_t> values) {
  absl::StatusOr<Tensor> tensor =
      Tensor::Create(DataType::DT_INT64, {(int64_t)values.size()},
                     CreateTestData<int64_t>(values));
  CHECK_OK(tensor.status());
  CHECK_OK(tensor->set_name(std::move(name)));
  return *std::move(tensor);
}

std::string BuildCheckpointFromTensors(std::vector<Tensor> tensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  for (const auto& tensor : tensors) {
    CHECK_OK(builder->Add(tensor.name(), tensor));
  }
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());
  return std::string(*checkpoint);
}

std::pair<BlobMetadata, std::string> EncryptWithKmsKeys(
    std::string message, std::string associated_data, std::string public_key) {
  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, public_key, associated_data);
  CHECK_OK(encrypt_result.status());

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypt_result->ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->set_ciphertext_associated_data(associated_data);
  encryption_metadata->set_encrypted_symmetric_key(
      encrypt_result->encrypted_symmetric_key);
  encryption_metadata->set_encapsulated_public_key(
      encrypt_result->encapped_key);
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(associated_data);
  return std::make_pair(metadata, encrypt_result->ciphertext);
}

// Creates a checkpoint with a privacy ID and event times.
std::string BuildCheckpoint(std::string privacy_id_val,
                            std::vector<std::string> event_times,
                            absl::string_view on_device_query_name) {
  std::vector<Tensor> tensors;

  Tensor privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {privacy_id_val});
  tensors.push_back(std::move(privacy_id_tensor));

  Tensor event_times_tensor = BuildStringTensor(
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName),
      event_times);
  tensors.push_back(std::move(event_times_tensor));
  return BuildCheckpointFromTensors(std::move(tensors));
}

// Creates an encrypted checkpoint with a privacy ID and event times.
std::pair<BlobMetadata, std::string> BuildEncryptedCheckpoint(
    std::string privacy_id_val, std::vector<std::string> event_times,
    std::string public_key, std::string associated_data,
    absl::string_view on_device_query_name) {
  std::string checkpoint =
      BuildCheckpoint(privacy_id_val, event_times, on_device_query_name);

  return EncryptWithKmsKeys(checkpoint, associated_data, public_key);
}

}  // namespace confidential_federated_compute::metadata::testing
