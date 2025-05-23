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

#include "containers/program_executor_tee/program_context/cc/fake_data_read_write_service.h"

#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidentialcompute::outgoing::WriteRequest;
using ::fcp::confidentialcompute::outgoing::WriteResponse;

grpc::Status FakeDataReadWriteService::Write(
    ::grpc::ServerContext*, ::grpc::ServerReader<WriteRequest>* request_reader,
    WriteResponse*) {
  // Append the stream of requests to write_call_args_.
  std::vector<WriteRequest> requests;
  WriteRequest request;
  while (request_reader->Read(&request)) {
    requests.push_back(request);
  }
  write_call_args_.push_back(requests);
  return grpc::Status::OK;
}

std::vector<std::vector<WriteRequest>>
FakeDataReadWriteService::GetWriteCallArgs() {
  return write_call_args_;
}

}  // namespace confidential_federated_compute::program_executor_tee