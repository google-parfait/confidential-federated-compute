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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::minimum_jax {

// Not thread safe
class SimpleSession final
    : public confidential_federated_compute::LegacySession {
 public:
  SimpleSession() {};

  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override {
    return absl::OkStatus();
  }
  // Adds a data blob from a given URI into the session and caches it.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override;
  // No-op
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override {
    return ToSessionCommitResponse(absl::OkStatus());
  }
  // Perform computation on all cached data.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override;

 private:
  std::vector<std::string> data_;
};

class SimpleConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  explicit SimpleConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle)
      : ConfidentialTransformBase(std::move(signing_key_handle)) {}

 protected:
  // No-op
  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    return google::protobuf::Struct();
  }
  // No-op
  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    return std::make_unique<SimpleSession>();
  }

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;
};

}  // namespace confidential_federated_compute::minimum_jax

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_