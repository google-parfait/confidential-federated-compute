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

#include "containers/batched_inference/batched_inference_server.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "containers/batched_inference/batched_inference_fn.h"
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/fns/confidential_transform_server.h"
#include "containers/fns/fn_factory.h"
#include "google/protobuf/any.pb.h"
#include "grpcpp/grpcpp.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::FnFactoryProvider;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::google::protobuf::Any;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::sdk::InstanceEncryptionKeyHandle;
using ::oak::containers::sdk::InstanceSigningKeyHandle;
using oak::crypto::EncryptionKeyHandle;
using oak::crypto::SigningKeyHandle;

// Increase gRPC message size limit to 2GB.
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;

FnFactoryProvider CreateBatchedInferenceFnFactoryProvider(
    std::shared_ptr<BatchedInferenceProvider> batched_inference_provider) {
  return [batched_inference_provider](
             const Any& configuration, const Any& config_constraints,
             const WriteConfigurationMap& write_configuration_map)
             -> absl::StatusOr<std::unique_ptr<FnFactory>> {
    return CreateBatchedInferenceFnFactory(batched_inference_provider);
  };
}

class BatchedInferenceServerImpl : public BatchedInferenceServer {
 public:
  BatchedInferenceServerImpl(
      std::unique_ptr<fns::FnConfidentialTransform> service,
      std::unique_ptr<Server> server, int port)
      : service_(std::move(service)), server_(std::move(server)), port_(port) {}

  virtual ~BatchedInferenceServerImpl() {
    server_->Shutdown();
    server_->Wait();
  }

  void Wait() override { server_->Wait(); }

  int port() override { return port_; }

 private:
  std::unique_ptr<fns::FnConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  int port_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<BatchedInferenceServer>>
CreateBatchedInferenceServer(
    std::shared_ptr<BatchedInferenceProvider> batched_inference_provider,
    int incoming_port, std::unique_ptr<SigningKeyHandle> signing_handle,
    std::unique_ptr<EncryptionKeyHandle> encryption_handle) {
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);

  std::string server_address = absl::StrFormat("[::]:%d", incoming_port);
  int selected_port;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);

  std::unique_ptr<fns::FnConfidentialTransform> service =
      std::make_unique<fns::FnConfidentialTransform>(
          std::move(signing_handle),
          CreateBatchedInferenceFnFactoryProvider(batched_inference_provider),
          std::move(encryption_handle));
  builder.RegisterService(service.get());

  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO)
      << "Batched Inference Confidential Transform Server listening on port "
      << selected_port;

  return std::unique_ptr<BatchedInferenceServer>(new BatchedInferenceServerImpl(
      std::move(service), std::move(server), selected_port));
}

}  // namespace confidential_federated_compute::batched_inference
