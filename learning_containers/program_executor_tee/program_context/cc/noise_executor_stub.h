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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_EXECUTOR_STUB_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_EXECUTOR_STUB_H_

#include "program_executor_tee/program_context/cc/noise_client_session.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Implementation of ExecutorGroup stub that transmits requests through
// a noise session and returns the responses. Only the synchronous methods are
// implemented.
class NoiseExecutorStub final
    : public tensorflow_federated::v0::ExecutorGroup::StubInterface {
 public:
  NoiseExecutorStub(NoiseClientSessionInterface* noise_client_session)
      : noise_client_session_(noise_client_session) {}

  ~NoiseExecutorStub() {}

  // Synchronous methods
  grpc::Status GetExecutor(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::GetExecutorRequest& request,
      tensorflow_federated::v0::GetExecutorResponse* response);
  grpc::Status CreateValue(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateValueRequest& request,
      tensorflow_federated::v0::CreateValueResponse* response);
  grpc::Status CreateCall(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateCallRequest& request,
      tensorflow_federated::v0::CreateCallResponse* response);
  grpc::Status CreateStruct(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateStructRequest& request,
      tensorflow_federated::v0::CreateStructResponse* response);
  grpc::Status CreateSelection(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateSelectionRequest& request,
      tensorflow_federated::v0::CreateSelectionResponse* response);
  grpc::Status Compute(grpc::ClientContext* context,
                       const tensorflow_federated::v0::ComputeRequest& request,
                       tensorflow_federated::v0::ComputeResponse* response);
  grpc::Status Dispose(grpc::ClientContext* context,
                       const tensorflow_federated::v0::DisposeRequest& request,
                       tensorflow_federated::v0::DisposeResponse* response);
  grpc::Status DisposeExecutor(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::DisposeExecutorRequest& request,
      tensorflow_federated::v0::DisposeExecutorResponse* response);

  // Asynchronous methods
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::GetExecutorResponse>*
  AsyncGetExecutorRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::GetExecutorRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::GetExecutorResponse>*
  PrepareAsyncGetExecutorRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::GetExecutorRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateValueResponse>*
  AsyncCreateValueRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateValueRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateValueResponse>*
  PrepareAsyncCreateValueRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateValueRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateCallResponse>*
  AsyncCreateCallRaw(grpc::ClientContext* context,
                     const tensorflow_federated::v0::CreateCallRequest& request,
                     grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateCallResponse>*
  PrepareAsyncCreateCallRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateCallRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateStructResponse>*
  AsyncCreateStructRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateStructRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateStructResponse>*
  PrepareAsyncCreateStructRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateStructRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateSelectionResponse>*
  AsyncCreateSelectionRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateSelectionRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::CreateSelectionResponse>*
  PrepareAsyncCreateSelectionRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::CreateSelectionRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::ComputeResponse>*
  AsyncComputeRaw(grpc::ClientContext* context,
                  const tensorflow_federated::v0::ComputeRequest& request,
                  grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::ComputeResponse>*
  PrepareAsyncComputeRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::ComputeRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::DisposeResponse>*
  AsyncDisposeRaw(grpc::ClientContext* context,
                  const tensorflow_federated::v0::DisposeRequest& request,
                  grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::DisposeResponse>*
  PrepareAsyncDisposeRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::DisposeRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::DisposeExecutorResponse>*
  AsyncDisposeExecutorRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::DisposeExecutorRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }
  grpc::ClientAsyncResponseReaderInterface<
      tensorflow_federated::v0::DisposeExecutorResponse>*
  PrepareAsyncDisposeExecutorRaw(
      grpc::ClientContext* context,
      const tensorflow_federated::v0::DisposeExecutorRequest& request,
      grpc::CompletionQueue* cq) {
    return nullptr;
  }

 private:
  NoiseClientSessionInterface* noise_client_session_;  // Not owned.
  absl::Mutex mutex_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_CLIENT_SESSION_H_
