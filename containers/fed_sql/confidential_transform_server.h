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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_

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
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

namespace confidential_federated_compute::fed_sql {

// ConfidentialTransform service for Federated SQL. Executes the aggregation
// step of FedSQL.
// TODO: execute the per-client SQL query step.
class FedSqlConfidentialTransform final
    : public fcp::confidentialcompute::ConfidentialTransform::Service {
 public:
  // The OrchestratorCrypto stub must not be NULL and must outlive this object.
  // TODO: add absl::Nonnull to crypto_stub.
  explicit FedSqlConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub,
      int max_num_sessions, long max_session_memory_bytes)
      : crypto_stub_(*ABSL_DIE_IF_NULL(crypto_stub)),
        session_tracker_(max_num_sessions, max_session_memory_bytes) {}

  grpc::Status Initialize(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::InitializeRequest* request,
      fcp::confidentialcompute::InitializeResponse* response) override;

  grpc::Status Session(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                               fcp::confidentialcompute::SessionRequest>*
          stream) override;

 private:
  absl::Status FedSqlInitialize(
      const fcp::confidentialcompute::InitializeRequest* request,
      fcp::confidentialcompute::InitializeResponse* response);

  absl::Status FedSqlSession(
      grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                               fcp::confidentialcompute::SessionRequest>*
          stream,
      long stream_memory);

  oak::containers::v1::OrchestratorCrypto::StubInterface& crypto_stub_;
  confidential_federated_compute::SessionTracker session_tracker_;
  absl::Mutex mutex_;
  // The mutex is used to protect the optional wrapping blob_decryptor_ and
  // intrinsics_ to ensure the BlobDecryptor and vector are initialized, but
  // the BlobDecryptor and const vector are themselves threadsafe.
  std::optional<const std::vector<tensorflow_federated::aggregation::Intrinsic>>
      intrinsics_ ABSL_GUARDED_BY(mutex_);
  std::optional<confidential_federated_compute::BlobDecryptor> blob_decryptor_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
