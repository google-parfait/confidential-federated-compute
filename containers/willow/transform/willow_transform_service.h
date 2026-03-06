// Copyright 2026 Google LLC.
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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_WILLOW_TRANSFORM_SERVICE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_WILLOW_TRANSFORM_SERVICE_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "willow/proto/willow/aggregation_config.pb.h"

namespace confidential_federated_compute::willow {

using SessionStream =
    grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                             fcp::confidentialcompute::SessionRequest>;

// Implementation of Willow specific Transform.
// TODO: refactor the implementation of the Service so that it derives
// from a new base class in the parent container that isn't KMS specific.
class WillowTransformService final
    : public fcp::confidentialcompute::ConfidentialTransform::Service {
 public:
  WillowTransformService() = default;

  grpc::Status StreamInitialize(
      grpc::ServerContext* context,
      grpc::ServerReader<fcp::confidentialcompute::StreamInitializeRequest>*
          reader,
      fcp::confidentialcompute::InitializeResponse* response) override;

  grpc::Status Session(grpc::ServerContext* context,
                       SessionStream* stream) override;

 private:
  absl::Status StreamInitializeImpl(
      grpc::ServerReader<fcp::confidentialcompute::StreamInitializeRequest>*
          reader,
      fcp::confidentialcompute::InitializeResponse* response);
  absl::Status SessionImpl(SessionStream* stream);
  absl::Status InitializeTransform(const google::protobuf::Any& configuration);
  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession();

  bool IsInitialized() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return session_tracker_.has_value();
  }

  // The mutex is used to protect the session_tracker_, to ensure it is
  // initialized, but SessionTracker is itself threadsafe.
  mutable absl::Mutex mutex_;
  std::optional<SessionTracker> session_tracker_ ABSL_GUARDED_BY(mutex_);

  // Aggregation configuration is initialized during StreamInitializeTransform.
  secure_aggregation::willow::AggregationConfigProto aggregation_config_;
};

}  // namespace confidential_federated_compute::willow

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_WILLOW_TRANSFORM_SERVICE_H_
