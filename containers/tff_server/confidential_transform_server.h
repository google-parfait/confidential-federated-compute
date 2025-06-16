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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_TFF_SERVER_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_TFF_SERVER_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::tff_server {

// TFF implementation of Session interface. Not threadsafe.
class TffSession final : public confidential_federated_compute::Session {
 public:
  TffSession() {};

  // Configure the session with the computation to be run, the initial
  // argument, and create the TFF executor with the specified number of clients.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override;
  // Adds a data blob from a given URI into the session and parses the data into
  // a TFF value.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override;
  // Currently no action taken for commits.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionCommit(
      const fcp::confidentialcompute::CommitRequest& commit_request) override {
    return ToSessionCommitResponse(absl::OkStatus());
  }
  // Resolves all data URIs in the initial argument, embeds the argument into
  // the TFF stack, executes the computation, and encrypts and outputs the
  // result.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override;

 private:
  absl::StatusOr<tensorflow_federated::v0::Value> FetchData(
      const std::string& uri);
  absl::StatusOr<tensorflow_federated::v0::Value> FetchClientData(
      const std::string& uri, const std::string& key);
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> ParseData(
      const std::string& uri, std::string unencrypted_data,
      int64_t total_size_bytes);
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> ParseClientData(
      const std::string& uri, std::string unencrypted_data,
      int64_t total_size_bytes);

  tensorflow_federated::v0::Value function_;
  std::optional<tensorflow_federated::v0::Value> argument_ = std::nullopt;
  uint32_t output_access_policy_node_id_;
  // TFF executor to which lambda computations can be delegated after
  // Data values have been resolved.
  std::shared_ptr<tensorflow_federated::Executor> child_executor_;
  // Map of URI to TFF Values that have been added into the session.
  absl::flat_hash_map<std::string, tensorflow_federated::v0::Value>
      data_by_uri_;
  // Map of URI to ClientCheckpointParsers that contain client data uploads that
  // have been added into the session.
  absl::flat_hash_map<
      std::string,
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
      client_checkpoint_parser_by_uri_;
};

// ConfidentialTransform service for Tensorflow Federated.
class TffConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  explicit TffConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle)
      : ConfidentialTransformBase(std::move(signing_key_handle)) {}

 protected:
  // No transform specific stream initialization for TFF.
  virtual absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    return google::protobuf::Struct();
  }
  // For TFF, ReadWriteConfigurationRequest is a no-op because there is no
  // initialization.
  virtual absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }

  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    return std::make_unique<TffSession>();
  }
};

}  // namespace confidential_federated_compute::tff_server

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_TFF_SERVER_CONFIDENTIAL_TRANSFORM_SERVER_H_
