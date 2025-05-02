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

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "containers/oak_orchestrator_client.h"
#include "containers/program_worker/program_worker_server.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "proto/containers/interfaces.grpc.pb.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"
#include "proto/session/session.pb.h"

namespace confidential_federated_compute::program_worker {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::google::protobuf::Empty;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::Orchestrator;
using ::oak::containers::v1::OrchestratorCrypto;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;

// Increase gRPC message size limit to 2GB
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;
static constexpr char kAttesterId[] = "attester_id";

void RunServer() {
  std::string server_address("[::]:8080");
  std::shared_ptr<grpc::Channel> orchestrator_channel =
      CreateOakOrchestratorChannel();

  OrchestratorCrypto::Stub orchestrator_crypto_stub(orchestrator_channel);

  Orchestrator::Stub orchestrator_stub(orchestrator_channel);

  grpc::ClientContext orchestrator_context;
  Empty empty_request;
  oak::session::v1::EndorsedEvidence endorsed_evidence;
  auto status = orchestrator_stub.GetEndorsedEvidence(
      &orchestrator_context, empty_request, &endorsed_evidence);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to get endorsed evidence. Orchestrator returned "
                  "error status: "
               << status.error_code() << ": " << status.error_message();
  }
  auto attester = bindings::new_simple_attester(ffi_bindings::BytesView(
      endorsed_evidence.evidence().SerializeAsString()));
  if (attester.error != nullptr) {
    LOG(FATAL) << "Failed to create attester:"
               << ffi_bindings::ErrorIntoStatus(attester.error);
  }
  auto endorser = bindings::new_simple_endorser(ffi_bindings::BytesView(
      endorsed_evidence.endorsements().SerializeAsString()));
  if (endorser.error != nullptr) {
    LOG(FATAL) << "Failed to create endorser:"
               << ffi_bindings::ErrorIntoStatus(endorser.error);
  }
  auto signing_key = bindings::new_random_signing_key();
  auto* session_config =
      SessionConfigBuilder(AttestationType::kSelfUnidirectional,
                           HandshakeType::kNoiseNN)
          .AddSelfAttester(kAttesterId, attester.result)
          .AddSelfEndorser(kAttesterId, endorser.result)
          .AddSessionBinder(kAttesterId, signing_key)
          .Build();
  bindings::free_signing_key(signing_key);

  auto service =
      ProgramWorkerTee::Create(&orchestrator_crypto_stub, session_config);
  CHECK_OK(service) << "Failed to create ProgramWorkerTee service: "
                    << service.status();

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service->get());
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Program Worker Server listening on " << server_address << "\n";

  OakOrchestratorClient oak_orchestrator_client(&orchestrator_stub);
  CHECK_OK(oak_orchestrator_client.NotifyAppReady());
  server->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::program_worker

int main(int argc, char** argv) {
  confidential_federated_compute::program_worker::RunServer();
  return 0;
}
