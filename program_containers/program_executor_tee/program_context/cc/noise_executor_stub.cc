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

#include "program_executor_tee/program_context/cc/noise_executor_stub.h"

#include "fcp/base/status_converters.h"
#include "program_executor_tee/proto/executor_wrapper.pb.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::base::ToGrpcStatus;
using ::oak::session::v1::PlaintextMessage;

grpc::Status NoiseExecutorStub::GetExecutor(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::GetExecutorRequest& request,
    tensorflow_federated::v0::GetExecutorResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_get_executor_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_get_executor_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected GetExecutorResponse.");
  }
  *response = result.get_executor_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::CreateValue(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::CreateValueRequest& request,
    tensorflow_federated::v0::CreateValueResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_create_value_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_create_value_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected GetCreateValueResponse.");
  }
  *response = result.create_value_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::CreateCall(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::CreateCallRequest& request,
    tensorflow_federated::v0::CreateCallResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_create_call_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_create_call_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected GetCreateCallResponse.");
  }
  *response = result.create_call_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::CreateStruct(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::CreateStructRequest& request,
    tensorflow_federated::v0::CreateStructResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_create_struct_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_create_struct_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected GetCreateStructResponse.");
  }
  *response = result.create_struct_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::CreateSelection(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::CreateSelectionRequest& request,
    tensorflow_federated::v0::CreateSelectionResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_create_selection_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_create_selection_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected GetCreateSelectionResponse.");
  }
  *response = result.create_selection_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::Compute(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::ComputeRequest& request,
    tensorflow_federated::v0::ComputeResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_compute_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_compute_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected ComputeResponse.");
  }
  *response = result.compute_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::Dispose(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::DisposeRequest& request,
    tensorflow_federated::v0::DisposeResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_dispose_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_dispose_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected DisposeResponse.");
  }
  *response = result.dispose_response();
  return grpc::Status::OK;
}

grpc::Status NoiseExecutorStub::DisposeExecutor(
    grpc::ClientContext* context,
    const tensorflow_federated::v0::DisposeExecutorRequest& request,
    tensorflow_federated::v0::DisposeExecutorResponse* response) {
  absl::MutexLock lock(&mutex_);

  executor_wrapper::ExecutorGroupRequest executor_group_request;
  *executor_group_request.mutable_dispose_executor_request() = request;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(executor_group_request.SerializeAsString());
  absl::StatusOr<PlaintextMessage> plaintext_response =
      noise_client_session_->DelegateComputation(plaintext_request);
  if (!plaintext_response.ok()) {
    return ToGrpcStatus(plaintext_response.status());
  }
  executor_wrapper::ExecutorGroupResponse result;
  if (!result.ParseFromString(plaintext_response->plaintext())) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Failed to parse response as ExecutorGroupResponse.");
  }
  if (!result.has_dispose_executor_response()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "Expected DisposeExecutorResponse.");
  }
  *response = result.dispose_executor_response();
  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_executor_tee
