
// Copyright 2023 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <memory>

#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::Channel;
using ::grpc::ClientContext;
using ::grpc::Status;

int main(int argc, char** argv) {
  std::cout << "Starting RPC Client for Pipeline Transform Server."
            << std::endl;

  std::shared_ptr<Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<PipelineTransform::Stub> stub_ =
      PipelineTransform::NewStub(channel);
  TransformRequest request;
  TransformResponse response;
  ClientContext context;

  Status status = stub_->Transform(&context, request, &response);
  if (status.ok()) {
    std::cout << "RPC success" << std::endl;
  } else {
    std::cout << "RPC failed: " << status.error_code() << ": "
              << status.error_message() << std::endl;
  }

  return 0;
}
