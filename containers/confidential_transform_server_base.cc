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

#include <cstdint>
#include <execution>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "cc/crypto/server_encryptor.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/nonce.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidential_compute::NonceChecker;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ServerContext;
using ::oak::crypto::DecryptionResult;
using ::oak::crypto::ServerEncryptor;

// Provides implementation of methods to emit the data into the session
// stream with an ability to encrypt and chunk the data.
class ChunkedStreamWriter : public Session::Context {
 public:
  ChunkedStreamWriter(SessionStream* stream, uint32_t chunk_size,
                      const std::optional<KmsEncryptor>& encryptor)
      : stream_(stream), chunk_size_(chunk_size), encryptor_(encryptor) {}

 private:
  // Implementation of Session::Context - private because these
  // methods can be called only via the Context interface.
  bool Emit(ReadResponse read_response) override;
  bool EmitUnencrypted(Session::KV kv) override;
  bool EmitEncrypted(int reencryption_key_index, Session::KV kv) override;

  bool Emit(std::string data, google::protobuf::Any config,
            BlobMetadata metadata);

  SessionStream* stream_;
  uint32_t chunk_size_;
  const std::optional<KmsEncryptor>& encryptor_;
};

bool ChunkedStreamWriter::Emit(ReadResponse read_response) {
  SessionResponse response;
  ReadResponse* mutable_read = response.mutable_read();
  *mutable_read = std::move(read_response);
  mutable_read->set_finish_read(mutable_read->data().size() <= chunk_size_);
  if (!mutable_read->finish_read()) {
    // Chunked write implemented by splitting the provided
    // ReadResponse.
    absl::Cord data(std::move(*mutable_read->mutable_data()));
    do {
      absl::CopyCordToString(data.Subcord(0, chunk_size_),
                             mutable_read->mutable_data());
      data.RemovePrefix(chunk_size_);
      if (!stream_->Write(response)) {
        return false;
      }
      mutable_read->clear_first_response_configuration();
      mutable_read->clear_first_response_metadata();
    } while (data.size() > chunk_size_);

    absl::CopyCordToString(data, mutable_read->mutable_data());
    mutable_read->set_finish_read(true);
  }

  // Final chunk of data (or the only chunk if the data was smaller or
  // equal than the chunk size).
  return stream_->Write(response);
}

bool ChunkedStreamWriter::Emit(std::string data, google::protobuf::Any key,
                               BlobMetadata metadata) {
  ReadResponse response;
  *response.mutable_data() = std::move(data);
  *response.mutable_first_response_configuration() = std::move(key);
  *response.mutable_first_response_metadata() = std::move(metadata);
  return Emit(std::move(response));
}

bool ChunkedStreamWriter::EmitUnencrypted(Session::KV kv) {
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(kv.data.size());
  metadata.mutable_unencrypted()->set_blob_id(kv.blob_id);
  return Emit(std::move(kv.data), std::move(kv.key), std::move(metadata));
}

bool ChunkedStreamWriter::EmitEncrypted(int reencryption_key_index,
                                        Session::KV kv) {
  if (!encryptor_.has_value()) {
    LOG(ERROR) << "KMS reencryption context isn't initialized";
    return false;
  }

  absl::StatusOr<KmsEncryptor::EncryptedResult> encrypted_result =
      encryptor_->EncryptIntermediateResult(reencryption_key_index, kv.data,
                                            kv.blob_id);
  if (!encrypted_result.ok()) {
    LOG(ERROR) << "Failed to encrypt intermediate result: "
               << encrypted_result.status();
    return false;
  }

  BlobMetadata metadata = std::move(encrypted_result->metadata);

  return Emit(std::move(encrypted_result->ciphertext), std::move(kv.key),
              std::move(metadata));
}

// Decrypts and parses a record and incorporates the record into the session.
//
// Reports status to the client in WriteFinishedResponse.
absl::Status ConfidentialTransformBase::HandleWrite(
    confidential_federated_compute::Session* session, WriteRequest request,
    absl::Cord blob_data, BlobDecryptor* blob_decryptor,
    std::optional<NonceChecker>& nonce_checker, SessionStream* stream,
    Session::Context& context) {
  if (nonce_checker.has_value()) {
    absl::Status nonce_status =
        nonce_checker->CheckBlobNonce(request.first_request_metadata());
    if (!nonce_status.ok()) {
      stream->Write(ToSessionWriteFinishedResponse(nonce_status));
      return absl::OkStatus();
    }
  }

  // Get the key ID if KMS is enabled. For legacy ledger, the key ID is not
  // needed.
  absl::StatusOr<std::string> key_id =
      KmsEnabled() ? GetKeyId(request.first_request_metadata()) : "";

  if (!key_id.ok()) {
    stream->Write(ToSessionWriteFinishedResponse(key_id.status()));
    return absl::OkStatus();
  }

  // TODO: Avoid flattening the cord, which requires the downstream
  // code to parse directly from the chunked cord.
  absl::StatusOr<std::string> unencrypted_data = blob_decryptor->DecryptBlob(
      request.first_request_metadata(), blob_data.Flatten(), key_id.value());
  if (!unencrypted_data.ok()) {
    LOG_EVERY_N(WARNING, 1000) << "Blob decryption failed for key_id "
                               << absl::BytesToHexString(key_id.value())
                               << " with status: " << unencrypted_data.status();
    stream->Write(ToSessionWriteFinishedResponse(unencrypted_data.status()));
    return absl::OkStatus();
  }
  // Clear the memory used by the original encrypted data.
  blob_data.Clear();

  SessionResponse response;
  FCP_ASSIGN_OR_RETURN(
      WriteFinishedResponse write_response,
      session->Write(std::move(request), std::move(unencrypted_data.value()),
                     context));

  *response.mutable_write() = std::move(write_response);
  stream->Write(response);
  return absl::OkStatus();
}

absl::Status ConfidentialTransformBase::StreamInitializeInternal(
    grpc::ServerReader<StreamInitializeRequest>* reader,
    InitializeResponse* response) {
  google::protobuf::Struct config_properties;
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  StreamInitializeRequest request;
  bool contain_initialize_request = false;
  uint32_t max_num_sessions;
  while (reader->Read(&request)) {
    switch (request.kind_case()) {
      case StreamInitializeRequest::kInitializeRequest: {
        // StreamInitializeRequest.initialize_request should be the last request
        // sent by the client's StreamInitializeRequest stream. Each stream
        // should only have exactly one
        // StreamInitializeRequest.initialize_request.
        if (contain_initialize_request) {
          return absl::FailedPreconditionError(
              "Expect one of the StreamInitializeRequests to be "
              "configured with a InitializeRequest, found more than one.");
        }
        const InitializeRequest& initialize_request =
            request.initialize_request();
        max_num_sessions = initialize_request.max_num_sessions();

        bool kms_enabled = initialize_request.has_protected_response() &&
                           oak_encryption_key_handle_ != nullptr;
        if (kms_enabled) {
          ServerEncryptor server_encryptor(*oak_encryption_key_handle_);
          FCP_ASSIGN_OR_RETURN(DecryptionResult decryption_result,
                               server_encryptor.Decrypt(
                                   initialize_request.protected_response()));
          if (!protected_response.ParseFromString(
                  decryption_result.plaintext)) {
            return absl::InvalidArgumentError(
                "Failed to parse ProtectedResponse from decrypted data.");
          }
          AuthorizeConfidentialTransformResponse::AssociatedData
              associated_data;
          if (!associated_data.ParseFromString(
                  decryption_result.associated_data)) {
            return absl::InvalidArgumentError(
                "Failed to parse AssociatedData from decrypted data.");
          }
          if (associated_data
                  .authorized_logical_pipeline_policies_hashes_size() == 0) {
            return absl::InvalidArgumentError(
                "Expected at least one policy hash but none were supplied.");
          }

          // Pick any of the authorized_logical_pipeline_policies_hashes as the
          // reencryption_policy_hash. For convenience, we pick the first one.
          kms_encryptor_.emplace(
              std::vector<std::string>(
                  protected_response.result_encryption_keys().begin(),
                  protected_response.result_encryption_keys().end()),
              associated_data.authorized_logical_pipeline_policies_hashes(0));
          for (const auto& policy_hash :
               associated_data.authorized_logical_pipeline_policies_hashes()) {
            authorized_logical_pipeline_policies_hashes_.insert(policy_hash);
          }
          FCP_RETURN_IF_ERROR(SetActiveKeyIds(
              {protected_response.decryption_keys().begin(),
               protected_response.decryption_keys().end()},
              {associated_data.omitted_decryption_key_ids().begin(),
               associated_data.omitted_decryption_key_ids().end()}));

          // TODO: stop passing reencryption context to the derived class and
          // instead use Session::Context to encrypt the outputs
          FCP_RETURN_IF_ERROR(StreamInitializeTransformWithKms(
              initialize_request.configuration(),
              associated_data.config_constraints(),
              kms_encryptor_->reencryption_keys(),
              kms_encryptor_->reencryption_policy_hash()));
        } else {
          FCP_ASSIGN_OR_RETURN(config_properties,
                               StreamInitializeTransform(&initialize_request));
        }
        contain_initialize_request = true;
        break;
      }
      case StreamInitializeRequest::kWriteConfiguration: {
        // Each stream should contain zero or more
        // StreamInitializeRequest.write_configurations, all of which must be
        // sent before the StreamInitializeRequest.initialize_request. The first
        // write_configuration for a blob will contain the metadata about the
        // blob, while the last will have a value of `commit` set to True.
        if (contain_initialize_request) {
          return absl::FailedPreconditionError(
              "Expect all StreamInitializeRequests.write_configurations to be "
              "sent before the StreamInitializeRequest.initialize_request.");
        }
        FCP_RETURN_IF_ERROR(
            ReadWriteConfigurationRequest(request.write_configuration()));
        break;
      }
      default:
        return absl::FailedPreconditionError(
            absl::StrCat("StreamInitializeRequest expected a "
                         "WriteConfigurationRequest or a "
                         "InitializeRequest, but received request of type: ",
                         request.kind_case()));
    }
  }
  if (!contain_initialize_request) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expect one of the StreamInitializeRequests to be configured with a "
        "InitializeRequest, found zero."));
  }

  const BlobDecryptor* blob_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (blob_decryptor_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "StreamInitialize can only be called once.");
    }
    blob_decryptor_.emplace(*oak_signing_key_handle_, config_properties,
                            std::vector<absl::string_view>(
                                protected_response.decryption_keys().begin(),
                                protected_response.decryption_keys().end()));
    session_tracker_.emplace(max_num_sessions);

    // Since blob_decryptor_ is set once in Initialize or StreamInitialize and
    // never modified, and the underlying object is threadsafe, it is safe to
    // store a local pointer to it and access the object without a lock after we
    // check under the mutex that a value has been set for the std::optional
    // wrapper.
    blob_decryptor = &*blob_decryptor_;
  }

  // Returning the public key is only needed with the legacy ledger.
  if (!KmsEnabled()) {
    FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                         blob_decryptor->GetPublicKey());
  }
  return absl::OkStatus();
}

absl::Status ConfidentialTransformBase::SetActiveKeyIds(
    const std::vector<absl::string_view>& decryption_keys,
    const std::vector<absl::string_view>& omitted_key_ids) {
  for (const auto& decryption_key : decryption_keys) {
    absl::StatusOr<OkpKey> key = OkpKey::Decode(decryption_key);
    if (!key.ok()) {
      LOG(WARNING) << "Skipping invalid decryption key: " << key.status();
      continue;
    }
    active_key_ids_.insert(key->key_id);
  }
  // The omitted_key_ids are still active but just not used for decryption by
  // this container.
  for (const auto& key_id : omitted_key_ids) {
    active_key_ids_.insert(std::string(key_id));
  }
  return absl::OkStatus();
}

absl::Status ConfidentialTransformBase::SessionImpl(SessionStream* stream) {
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
  uint32_t chunk_size = configure_request.configure().chunk_size();
  if (chunk_size == 0) {
    return absl::FailedPreconditionError(
        "chunk_size must be specified in the session ConfigureRequest.");
  }

  // This provides the context for encryption and writing of ReadResponse
  // messages.
  ChunkedStreamWriter context(stream, chunk_size, kms_encryptor_);

  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<confidential_federated_compute::Session> session,
      CreateSession());
  FCP_ASSIGN_OR_RETURN(
      ConfigureResponse configure_response,
      session->Configure(std::move(*configure_request.mutable_configure()),
                         context));
  SessionResponse response;
  std::optional<NonceChecker> nonce_checker = std::nullopt;
  // Nonces only need to be verified for the legacy ledger.
  if (!KmsEnabled()) {
    nonce_checker = NonceChecker();
    *configure_response.mutable_nonce() = nonce_checker->GetSessionNonce();
  }
  *response.mutable_configure() = std::move(configure_response);
  stream->Write(response);

  // Initialize result_blob_metadata with unencrypted metadata since
  // EarliestExpirationTimeMetadata expects inputs to have either unencrypted or
  // hpke_plus_aead_data.
  BlobMetadata result_blob_metadata;
  result_blob_metadata.mutable_unencrypted();
  SessionRequest session_request;

  // Describes in-progress write state when it arrives in multiple chunks.
  struct WriteState {
    // The request from the first chunk containing metadata and configuration
    // but with emptied out data field.
    WriteRequest first_request;
    // The data combined from all chunks.
    absl::Cord data;
  };
  std::optional<WriteState> write_state = std::nullopt;

  while (stream->Read(&session_request)) {
    switch (session_request.kind_case()) {
      case SessionRequest::kWrite: {
        WriteRequest& write_request = *session_request.mutable_write();
        bool is_commit = write_request.commit();
        if (write_state.has_value()) {
          // This is a continuation of an in-progress chunked blob write.
          // This request isn't supposed to have metadata.
          if (write_request.has_first_request_metadata()) {
            return absl::FailedPreconditionError(
                "Session expected a continuation of chunked blob Write request "
                "but received a Write request for another blob");
          }
          // Append the chunk.
          write_state->data.Append(std::move(*write_request.mutable_data()));
        } else {
          // This is a write request for a new blob i.e. the first chunk of a
          // multi-chunk blob or a small blob consisting of a single chunk of
          // data.

          // Use the metadata with the earliest expiration timestamp for
          // encrypting any results. This is only needed for the legacy ledger.
          // With KMS, the re-encryption keys are shared upfront with the worker
          // as part of initialization.
          if (!KmsEnabled()) {
            absl::StatusOr<BlobMetadata> earliest_expiration_metadata =
                EarliestExpirationTimeMetadata(
                    result_blob_metadata,
                    write_request.first_request_metadata());
            if (!earliest_expiration_metadata.ok()) {
              stream->Write(ToSessionWriteFinishedResponse(absl::Status(
                  earliest_expiration_metadata.status().code(),
                  absl::StrCat(
                      "Failed to extract expiration timestamp from "
                      "BlobMetadata with encryption: ",
                      earliest_expiration_metadata.status().message()))));
              break;
            }
            std::swap(result_blob_metadata, *earliest_expiration_metadata);
          }
          absl::Cord data(std::move(*write_request.mutable_data()));
          write_state = WriteState{.first_request = std::move(write_request),
                                   .data = std::move(data)};
        }
        if (is_commit) {
          FCP_RETURN_IF_ERROR(
              HandleWrite(session.get(), std::move(write_state->first_request),
                          std::move(write_state->data), blob_decryptor,
                          nonce_checker, stream, context));
          write_state.reset();
        }
        break;
      }

      case SessionRequest::kCommit: {
        if (write_state.has_value()) {
          return absl::FailedPreconditionError(
              "Session expected a continuation of chunked blob Write request "
              "but received a Commit request");
        }
        const CommitRequest& commit_request = session_request.commit();
        FCP_ASSIGN_OR_RETURN(CommitResponse commit_response,
                             session->Commit(commit_request, context));
        SessionResponse response;
        *response.mutable_commit() = std::move(commit_response);
        stream->Write(response);
        break;
      }

      case SessionRequest::kFinalize: {
        if (write_state.has_value()) {
          return absl::FailedPreconditionError(
              "Session expected a continuation of chunked blob Write request "
              "but received a Finalize request");
        }
        FCP_ASSIGN_OR_RETURN(FinalizeResponse finalize_response,
                             session->Finalize(session_request.finalize(),
                                               result_blob_metadata, context));
        SessionResponse response;
        *response.mutable_finalize() = std::move(finalize_response);
        stream->Write(response);
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

grpc::Status ConfidentialTransformBase::StreamInitialize(
    ServerContext* context, grpc::ServerReader<StreamInitializeRequest>* reader,
    InitializeResponse* response) {
  return ToGrpcStatus(StreamInitializeInternal(reader, response));
}

grpc::Status ConfidentialTransformBase::Session(ServerContext* context,
                                                SessionStream* stream) {
  SessionTracker* session_tracker;
  {
    absl::MutexLock l(&mutex_);
    if (session_tracker_ == std::nullopt) {
      return ToGrpcStatus(absl::FailedPreconditionError(
          "StreamInitialize must be called before Session."));
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
  grpc::Status status = ToGrpcStatus(SessionImpl(stream));
  absl::Status remove_session = session_tracker->RemoveSession();
  if (!remove_session.ok()) {
    return ToGrpcStatus(remove_session);
  }
  return status;
}

absl::StatusOr<BlobDecryptor*> ConfidentialTransformBase::GetBlobDecryptor() {
  absl::MutexLock l(&mutex_);
  if (blob_decryptor_ == std::nullopt) {
    return absl::FailedPreconditionError(
        "Initialize must be called before GetBlobDecryptor.");
  }

  // Since blob_decryptor_ is set once in Initialize and never modified,
  // and the underlying object is threadsafe, it is safe to store a local
  // pointer to it and access the object without a lock after we check under
  // the mutex that values have been set for the std::optional wrappers.
  return &*blob_decryptor_;
}

}  // namespace confidential_federated_compute
