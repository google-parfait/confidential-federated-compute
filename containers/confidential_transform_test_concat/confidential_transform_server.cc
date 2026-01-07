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

#include "containers/confidential_transform_test_concat/confidential_transform_server.h"

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "cc/crypto/encryption_key.h"
#include "cc/crypto/signing_key.h"
#include "containers/blob_metadata.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::confidential_transform_test_concat {

// TestConcat implementation of Session interface. Not threadsafe.
class TestConcatSession final : public confidential_federated_compute::Session {
 public:
  TestConcatSession() = default;

  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest request,
      Context& context) override {
    return fcp::confidentialcompute::ConfigureResponse{};
  }

  // Concatenates the unencrypted data to the result string.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest request,
      std::string unencrypted_data, Context& context) override {
    absl::StrAppend(&state_, unencrypted_data);
    return confidential_federated_compute::ToWriteFinishedResponse(
        absl::OkStatus(), request.first_request_metadata().total_size_bytes());
  }

  // Currently no action taken for commits.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest request,
      Context& context) override {
    return ToCommitResponse(absl::OkStatus());
  }

  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override {
    context.EmitUnencrypted(state_);
    return fcp::confidentialcompute::FinalizeResponse{};
  }

 private:
  std::string state_ = "";
};

TestConcatConfidentialTransform::TestConcatConfidentialTransform(
    std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
    std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
    : ConfidentialTransformBase(std::move(signing_key_handle),
                                std::move(encryption_key_handle)) {};

absl::Status TestConcatConfidentialTransform::StreamInitializeTransformWithKms(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    std::vector<std::string> reencryption_keys,
    absl::string_view reencryption_policy_hash) {
  return absl::OkStatus();
}

absl::Status TestConcatConfidentialTransform::ReadWriteConfigurationRequest(
    const fcp::confidentialcompute::WriteConfigurationRequest&
        write_configuration) {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
TestConcatConfidentialTransform::CreateSession() {
  return std::make_unique<
      confidential_federated_compute::confidential_transform_test_concat::
          TestConcatSession>();
};

absl::StatusOr<std::string> TestConcatConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  return GetKeyIdFromMetadata(metadata);
}

}  // namespace
   // confidential_federated_compute::confidential_transform_test_concat
