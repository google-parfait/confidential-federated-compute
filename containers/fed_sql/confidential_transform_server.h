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
#include "containers/confidential_transform_server_base.h"
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
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  FedSqlConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub,
      int max_num_sessions)
      : ConfidentialTransformBase(crypto_stub, max_num_sessions) {};

 protected:
  virtual absl::StatusOr<google::protobuf::Struct> InitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override;
  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;

 private:
  absl::Mutex mutex_;
  std::optional<const std::vector<tensorflow_federated::aggregation::Intrinsic>>
      intrinsics_ ABSL_GUARDED_BY(mutex_);
};

// FedSql implementation of Session interface. Not threadsafe.
class FedSqlSession final : public confidential_federated_compute::Session {
 public:
  FedSqlSession(
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
          aggregator,
      const std::vector<tensorflow_federated::aggregation::Intrinsic>&
          intrinsics)
      : aggregator_(std::move(aggregator)), intrinsics_(intrinsics) {};
  // Currently no FedSql per-session configuration.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override {
    return absl::OkStatus();
  }
  // Accumulates a record into the state of the CheckpointAggregator
  // `aggregator`.
  //
  // Returns an error if the aggcore state may be invalid and the session
  // needs to be shut down.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override;
  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override;

 private:
  // The aggregator used during the session to accumulate writes.
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
      aggregator_;
  const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_CONFIDENTIAL_TRANSFORM_SERVER_H_
