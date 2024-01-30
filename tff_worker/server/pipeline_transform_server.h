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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_TFF_WORKER_SERVER_PIPELINE_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_TFF_WORKER_SERVER_PIPELINE_TRANSFORM_SERVER_H_

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_worker_configuration.pb.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute::tff_worker {

class TffPipelineTransform final
    : public fcp::confidentialcompute::PipelineTransform::Service {
 public:
  grpc::Status ConfigureAndAttest(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response) override;

 private:
  absl::Status TffConfigureAndAttest(
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response);

  absl::Mutex mutex_;
  fcp::confidentialcompute::TffWorkerConfiguration
      tff_worker_configuration_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace confidential_federated_compute::tff_worker

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_TFF_WORKER_SERVER_PIPELINE_TRANSFORM_SERVER_H_
