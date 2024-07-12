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
#include "containers/confidential_transform_test_concat/confidential_transform_server.h"

#include <execution>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace confidential_federated_compute::confidential_transform_test_concat {

namespace {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidential_compute::NonceChecker;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ServerContext;

}  // namespace

absl::Status TestConcatConfidentialTransform::Initialize(
    const fcp::confidentialcompute::InitializeRequest* request,
    fcp::confidentialcompute::InitializeResponse* response) {
  const BlobDecryptor* blob_decryptor;
  {
    absl::MutexLock l(&mutex_);
    blob_decryptor_.emplace(crypto_stub_);

    // Since blob_decryptor_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    blob_decryptor = &*blob_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                       blob_decryptor->GetPublicKey());
  return absl::OkStatus();
}

absl::Status TestConcatConfidentialTransform::Session(
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  BlobDecryptor* blob_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (blob_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }

    // Since blob_decryptor_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    blob_decryptor = &*blob_decryptor_;
  }

  SessionRequest configure_request;
  bool success = stream->Read(&configure_request);
  if (!success) {
    return absl::AbortedError("Session failed to read client message.");
  }

  if (!configure_request.has_configure()) {
    return absl::FailedPreconditionError(
        "Session must be configured with a ConfigureRequest before any other "
        "requests.");
  }
  SessionResponse configure_response;
  NonceChecker nonce_checker;
  *configure_response.mutable_configure()->mutable_nonce() =
      nonce_checker.GetSessionNonce();
  configure_response.mutable_configure();
  stream->Write(configure_response);

  SessionRequest session_request;
  std::string state = "";
  while (stream->Read(&session_request)) {
    switch (session_request.kind_case()) {
      case SessionRequest::kWrite: {
        const WriteRequest& write_request = session_request.write();
        if (absl::Status nonce_status = nonce_checker.CheckBlobNonce(
                write_request.first_request_metadata());
            !nonce_status.ok()) {
          stream->Write(ToSessionWriteFinishedResponse(nonce_status,
                                                       /*available_memory*/ 0));
          break;
        }

        absl::StatusOr<std::string> unencrypted_data =
            blob_decryptor->DecryptBlob(write_request.first_request_metadata(),
                                        write_request.data());
        if (!unencrypted_data.ok()) {
          stream->Write(
              ToSessionWriteFinishedResponse(unencrypted_data.status(),
                                             /*available_memory*/ 0));
          break;
        }

        absl::StrAppend(&state, *unencrypted_data);
        stream->Write(ToSessionWriteFinishedResponse(
            absl::OkStatus(), /*available_memory*/ 0,
            write_request.first_request_metadata().total_size_bytes()));
        break;
      }
      case SessionRequest::kFinalize: {
        SessionResponse response;
        ReadResponse* read_response = response.mutable_read();
        read_response->set_finish_read(true);
        *(read_response->mutable_data()) = state;

        BlobMetadata result_metadata;
        result_metadata.mutable_unencrypted();
        result_metadata.set_total_size_bytes(state.length());
        result_metadata.set_compression_type(
            BlobMetadata::COMPRESSION_TYPE_NONE);
        *(read_response->mutable_first_response_metadata()) = result_metadata;

        stream->Write(response);
        return absl::OkStatus();
      }
      case SessionRequest::kConfigure:
      default:
        return absl::FailedPreconditionError(
            absl::StrCat("Session expected a write or finalize request but "
                         "received request of type: ",
                         session_request.kind_case()));
    }
  }

  return absl::AbortedError(
      "Session failed to read client write or finalize message.");
}

grpc::Status TestConcatConfidentialTransform::Initialize(
    ServerContext* context, const InitializeRequest* request,
    InitializeResponse* response) {
  return ToGrpcStatus(Initialize(request, response));
}

grpc::Status TestConcatConfidentialTransform::Session(
    ServerContext* context,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  grpc::Status status = ToGrpcStatus(Session(stream));
  return status;
}

}  // namespace
   // confidential_federated_compute::confidential_transform_test_concat
