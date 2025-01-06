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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_TEST_CONCAT_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_TEST_CONCAT_CONFIDENTIAL_TRANSFORM_SERVER_H_

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
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute::confidential_transform_test_concat {

// TestConcat implementation of Session interface. Not threadsafe.
class TestConcatSession final : public confidential_federated_compute::Session {
 public:
  TestConcatSession() {};
  // Currently no per-session configuration.
  absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) override {
    return absl::OkStatus();
  }
  // Concatenates the unencrypted data to the result string.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> SessionWrite(
      const fcp::confidentialcompute::WriteRequest& write_request,
      std::string unencrypted_data) override {
    absl::StrAppend(&state_, unencrypted_data);
    return confidential_federated_compute::ToSessionWriteFinishedResponse(
        absl::OkStatus(),
        write_request.first_request_metadata().total_size_bytes());
  }
  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::SessionResponse> FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) override {
    fcp::confidentialcompute::SessionResponse response;
    fcp::confidentialcompute::ReadResponse* read_response =
        response.mutable_read();
    read_response->set_finish_read(true);
    *(read_response->mutable_data()) = state_;

    fcp::confidentialcompute::BlobMetadata result_metadata;
    result_metadata.mutable_unencrypted();
    result_metadata.set_total_size_bytes(state_.length());
    result_metadata.set_compression_type(
        fcp::confidentialcompute::BlobMetadata::COMPRESSION_TYPE_NONE);
    *(read_response->mutable_first_response_metadata()) = result_metadata;
    return response;
  }

 private:
  std::string state_ = "";
};

// Test ConfidentialTransform service that concatenates inputs.
class TestConcatConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  TestConcatConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
      : ConfidentialTransformBase(crypto_stub) {};

 protected:
  virtual absl::StatusOr<google::protobuf::Struct> InitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    google::protobuf::Struct config_properties;
    return config_properties;
  }
  virtual absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      fcp::confidentialcompute::StreamInitializeRequest* request) override {
    google::protobuf::Struct config_properties;
    return config_properties;
  }
  virtual absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }
  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    return std::make_unique<
        confidential_federated_compute::confidential_transform_test_concat::
            TestConcatSession>();
  };
};

}  // namespace
   // confidential_federated_compute::confidential_transform_test_concat

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_TEST_CONCAT_CONFIDENTIAL_TRANSFORM_SERVER_H_
