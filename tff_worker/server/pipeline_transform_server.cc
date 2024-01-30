/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tff_worker/server/pipeline_transform_server.h"

#include "absl/synchronization/mutex.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_worker_configuration.pb.h"
#include "grpcpp/server_context.h"

namespace confidential_federated_compute::tff_worker {


using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::grpc::ServerContext;


absl::Status TffPipelineTransform::TffConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  absl::MutexLock l(&mutex_);
  // Equating a 0-sized proto with an empty proto.
  if (tff_worker_configuration_.ByteSizeLong() != 0) {
    return absl::FailedPreconditionError(
        "ConfigureAndAttest can only be called once.");
  }
  if (!request->has_configuration()) {
    return absl::InvalidArgumentError(
        "ConfigureAndAttestRequest must contain configuration.");
  }

  if (!request->configuration().UnpackTo(&tff_worker_configuration_)) {
    return absl::InvalidArgumentError(
        "ConfigureAndAttestRequest configuration must be a "
        "tff_worker_configuration_pb2.TffWorkerConfiguration.");
  }

  // TODO: When encryption is implemented, this should cause generation of a new
  // keypair.
  return absl::OkStatus();
}

grpc::Status TffPipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(TffConfigureAndAttest(request, response));
}

}  // namespace confidential_federated_compute::tff_worker
