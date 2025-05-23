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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_

#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute::program_executor_tee {

// Fake DataReadWrite service that records WriteRequest call args.
class FakeDataReadWriteService
    : public fcp::confidentialcompute::outgoing::DataReadWrite::Service {
 public:
  grpc::Status Write(
      ::grpc::ServerContext*,
      ::grpc::ServerReader<fcp::confidentialcompute::outgoing::WriteRequest>*
          request_reader,
      fcp::confidentialcompute::outgoing::WriteResponse*) override;

  // Returns a list of received WriteRequest args.
  std::vector<std::vector<fcp::confidentialcompute::outgoing::WriteRequest>>
  GetWriteCallArgs();

 private:
  // List of received WriteRequest args.
  std::vector<std::vector<fcp::confidentialcompute::outgoing::WriteRequest>>
      write_call_args_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_