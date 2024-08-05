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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute {

// Base class that implements the ConfidentialTransform service protocol.
class ConfidentialTransformBase
    : public fcp::confidentialcompute::ConfidentialTransform::Service {
 public:
  grpc::Status Initialize(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::InitializeRequest* request,
      fcp::confidentialcompute::InitializeResponse* response) override;

  grpc::Status Session(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                               fcp::confidentialcompute::SessionRequest>*
          stream) override;

 protected:
  ConfidentialTransformBase(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
      : crypto_stub_(*ABSL_DIE_IF_NULL(crypto_stub)) {}

  virtual absl::StatusOr<google::protobuf::Struct> InitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) = 0;
  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() = 0;

 private:
  absl::Status InitializeInternal(
      const fcp::confidentialcompute::InitializeRequest* request,
      fcp::confidentialcompute::InitializeResponse* response);

  absl::Status SessionInternal(
      grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                               fcp::confidentialcompute::SessionRequest>*
          stream);

  oak::containers::v1::OrchestratorCrypto::StubInterface& crypto_stub_;
  absl::Mutex mutex_;
  // The mutex is used to protect the optional wrapping blob_decryptor_ and
  // session_tracker_ to ensure they are initialized, but the BlobDecryptor and
  // SessionTracker are themselves threadsafe.
  std::optional<confidential_federated_compute::BlobDecryptor> blob_decryptor_
      ABSL_GUARDED_BY(mutex_);
  std::optional<confidential_federated_compute::SessionTracker> session_tracker_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_
