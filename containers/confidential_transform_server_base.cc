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
#include "containers/confidential_transform_server_base.h"

#include <execution>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidential_compute::NonceChecker;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ServerContext;

namespace {

// Decrypts and parses a record and incorporates the record into the session.
//
// Reports status to the client in WriteFinishedResponse.
//
// TODO: handle blobs that span multiple WriteRequests.
absl::Status HandleWrite(
    const WriteRequest& request, BlobDecryptor* blob_decryptor,
    NonceChecker& nonce_checker,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream,
    Session* session) {
  if (absl::Status nonce_status =
          nonce_checker.CheckBlobNonce(request.first_request_metadata());
      !nonce_status.ok()) {
    stream->Write(ToSessionWriteFinishedResponse(nonce_status));
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> unencrypted_data = blob_decryptor->DecryptBlob(
      request.first_request_metadata(), request.data());
  if (!unencrypted_data.ok()) {
    stream->Write(ToSessionWriteFinishedResponse(unencrypted_data.status()));
    return absl::OkStatus();
  }

  FCP_ASSIGN_OR_RETURN(
      SessionResponse response,
      session->SessionWrite(request, std::move(unencrypted_data.value())));

  stream->Write(response);
  return absl::OkStatus();
}

}  // namespace

absl::Status ConfidentialTransformBase::InitializeInternal(
    const fcp::confidentialcompute::InitializeRequest* request,
    fcp::confidentialcompute::InitializeResponse* response) {
  FCP_ASSIGN_OR_RETURN(google::protobuf::Struct config_properties,
                       InitializeTransform(request));
  const BlobDecryptor* blob_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (blob_decryptor_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize can only be called once.");
    }
    blob_decryptor_.emplace(crypto_stub_, config_properties);
    session_tracker_.emplace(request->max_num_sessions());

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

absl::Status ConfidentialTransformBase::SessionInternal(
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
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<confidential_federated_compute::Session> session,
      CreateSession());
  FCP_RETURN_IF_ERROR(session->ConfigureSession(configure_request));
  SessionResponse configure_response;
  NonceChecker nonce_checker;
  *configure_response.mutable_configure()->mutable_nonce() =
      nonce_checker.GetSessionNonce();
  stream->Write(configure_response);

  // Initialze result_blob_metadata with unencrypted metadata since
  // EarliestExpirationTimeMetadata expects inputs to have either unencrypted or
  // hpke_plus_aead_data.
  BlobMetadata result_blob_metadata;
  result_blob_metadata.mutable_unencrypted();
  SessionRequest session_request;
  while (stream->Read(&session_request)) {
    switch (session_request.kind_case()) {
      case SessionRequest::kWrite: {
        const WriteRequest& write_request = session_request.write();
        // Use the metadata with the earliest expiration timestamp for
        // encrypting any results.
        absl::StatusOr<BlobMetadata> earliest_expiration_metadata =
            EarliestExpirationTimeMetadata(
                result_blob_metadata, write_request.first_request_metadata());
        if (!earliest_expiration_metadata.ok()) {
          stream->Write(ToSessionWriteFinishedResponse(absl::Status(
              earliest_expiration_metadata.status().code(),
              absl::StrCat("Failed to extract expiration timestamp from "
                           "BlobMetadata with encryption: ",
                           earliest_expiration_metadata.status().message()))));
          break;
        }
        result_blob_metadata = *earliest_expiration_metadata;
        // TODO: spin up a thread to incorporate each blob.
        FCP_RETURN_IF_ERROR(HandleWrite(write_request, blob_decryptor,
                                        nonce_checker, stream, session.get()));
        break;
      }
      case SessionRequest::kFinalize: {
        FCP_ASSIGN_OR_RETURN(
            SessionResponse finalize_response,
            session->FinalizeSession(session_request.finalize(),
                                     result_blob_metadata));
        stream->Write(finalize_response);
        return absl::OkStatus();
      }
      case SessionRequest::kConfigure:
      default:
        return absl::FailedPreconditionError(absl::StrCat(
            "Session expected a write request but received request of type: ",
            session_request.kind_case()));
    }
  }

  return absl::AbortedError(
      "Session failed to read client write or finalize message.");
}

grpc::Status ConfidentialTransformBase::Initialize(
    ServerContext* context, const InitializeRequest* request,
    InitializeResponse* response) {
  return ToGrpcStatus(InitializeInternal(request, response));
}

grpc::Status ConfidentialTransformBase::Session(
    ServerContext* context,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  SessionTracker* session_tracker;
  {
    absl::MutexLock l(&mutex_);
    if (session_tracker_ == std::nullopt) {
      return ToGrpcStatus(absl::FailedPreconditionError(
          "Initialize must be called before Session."));
    }

    // Since session_tracker_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    session_tracker = &*session_tracker_;
  }
  if (absl::Status session_status = session_tracker->AddSession();
      !session_status.ok()) {
    return ToGrpcStatus(session_status);
  }
  grpc::Status status = ToGrpcStatus(SessionInternal(stream));
  absl::Status remove_session = session_tracker->RemoveSession();
  if (!remove_session.ok()) {
    return ToGrpcStatus(remove_session);
  }
  return status;
}

}  // namespace confidential_federated_compute
