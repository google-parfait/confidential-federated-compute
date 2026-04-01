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
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureRequest;
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
// stream with an ability to encrypt and chunk the data. This class is not
// threadsafe.
class SessionContextImpl : public Session::Context {
 public:
  SessionContextImpl(SessionStream* stream,
                     const std::optional<KmsEncryptor>& encryptor)
      : stream_(stream), encryptor_(encryptor) {}

  void ApplyCountersToResponse(SessionResponse* response) {
    if (!counters_.empty()) {
      response->mutable_metrics()->mutable_counters()->insert(counters_.begin(),
                                                              counters_.end());
      counters_.clear();
    }
  }

 private:
  // Implementation of Session::Context - private because these
  // methods can be called only via the Context interface.
  bool Emit(ReadResponse read_response) override;
  bool EmitUnencrypted(Session::KV kv) override;
  bool EmitEncrypted(int reencryption_key_index, Session::KV kv) override;
  bool EmitReleasable(int reencryption_key_index, Session::KV kv,
                      std::optional<absl::string_view> src_state,
                      absl::string_view dst_state,
                      std::string& release_token) override;
  Counters& GetCounters() override;

  bool Emit(std::string data, google::protobuf::Any config,
            BlobMetadata metadata);

  SessionStream* stream_;
  const std::optional<KmsEncryptor>& encryptor_;
  Counters counters_;
};

Counters& SessionContextImpl::GetCounters() { return counters_; }

bool SessionContextImpl::Emit(ReadResponse read_response) {
  SessionResponse session_response;
  *session_response.mutable_read() = std::move(read_response);
  if (auto status = stream_->Write(session_response); !status.ok()) {
    LOG(ERROR) << "Failed to write ReadResponse to SessionStream: " << status;
    return false;
  }
  return true;
}

bool SessionContextImpl::Emit(std::string data, google::protobuf::Any key,
                              BlobMetadata metadata) {
  ReadResponse response;
  response.set_data(std::move(data));
  *response.mutable_first_response_configuration() = std::move(key);
  *response.mutable_first_response_metadata() = std::move(metadata);
  return Emit(std::move(response));
}

bool SessionContextImpl::EmitUnencrypted(Session::KV kv) {
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(kv.data.size());
  metadata.mutable_unencrypted()->set_blob_id(kv.blob_id);
  return Emit(std::move(kv.data), std::move(kv.key), std::move(metadata));
}

bool SessionContextImpl::EmitEncrypted(int reencryption_key_index,
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

bool SessionContextImpl::EmitReleasable(
    int reencryption_key_index, Session::KV kv,
    std::optional<absl::string_view> src_state, absl::string_view dst_state,
    std::string& release_token) {
  if (!encryptor_.has_value()) {
    LOG(ERROR) << "KMS reencryption context isn't initialized";
    return false;
  }

  absl::StatusOr<KmsEncryptor::EncryptedResult> encrypted_result =
      encryptor_->EncryptReleasableResult(reencryption_key_index, kv.data,
                                          kv.blob_id, src_state, dst_state);

  if (!encrypted_result.ok()) {
    LOG(ERROR) << "Failed to encrypt releasable result: "
               << encrypted_result.status();
    return false;
  }

  if (!encrypted_result->release_token.has_value()) {
    LOG(ERROR) << "EncryptReleasableResult did not return a release token";
    return false;
  }
  release_token = std::move(encrypted_result->release_token.value());

  BlobMetadata metadata = std::move(encrypted_result->metadata);

  return Emit(std::move(encrypted_result->ciphertext), std::move(kv.key),
              std::move(metadata));
}

// Decrypts and parses a record and incorporates the record into the session.
//
// Returns a WriteFinishedResponse with the status of the operation.
absl::StatusOr<WriteFinishedResponse> ConfidentialTransformBase::HandleWrite(
    confidential_federated_compute::Session* session, WriteRequest request,
    Decryptor* decryptor, Session::Context& context) {
  // Get the key ID from the metadata.
  absl::StatusOr<std::string> key_id =
      GetKeyId(request.first_request_metadata());
  if (!key_id.ok()) {
    return ToWriteFinishedResponse(key_id.status());
  }

  // TODO: Avoid flattening the cord, which requires the downstream
  // code to parse directly from the chunked cord.
  absl::Cord data = request.data();
  // Clear the memory used by the original encrypted data.
  request.set_data("");

  absl::StatusOr<std::string> unencrypted_data = decryptor->DecryptBlob(
      request.first_request_metadata(), data.Flatten(), key_id.value());
  data.Clear();
  if (!unencrypted_data.ok()) {
    LOG_EVERY_N(WARNING, 1000) << "Blob decryption failed for key_id "
                               << absl::BytesToHexString(key_id.value())
                               << " with status: " << unencrypted_data.status();
    return ToWriteFinishedResponse(unencrypted_data.status());
  }

  return session->Write(std::move(request), std::move(unencrypted_data.value()),
                        context);
}

absl::Status ConfidentialTransformBase::StreamInitializeInternal(
    grpc::ServerReader<StreamInitializeRequest>* reader,
    InitializeResponse* response) {
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
        ServerEncryptor server_encryptor(*oak_encryption_key_handle_);
        FCP_ASSIGN_OR_RETURN(
            DecryptionResult decryption_result,
            server_encryptor.Decrypt(initialize_request.protected_response()));
        if (!protected_response.ParseFromString(decryption_result.plaintext)) {
          return absl::InvalidArgumentError(
              "Failed to parse ProtectedResponse from decrypted data.");
        }
        AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
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
        active_key_ids_include_all_keysets_ =
            associated_data.omitted_decryption_key_ids_include_all_keysets();
        // Pick any of the authorized_logical_pipeline_policies_hashes as the
        // reencryption_policy_hash. For convenience, we pick the first one.
        kms_encryptor_.emplace(
            std::vector<std::string>(
                protected_response.result_encryption_keys().begin(),
                protected_response.result_encryption_keys().end()),
            associated_data.authorized_logical_pipeline_policies_hashes(0),
            oak_signing_key_handle_);
        for (const auto& policy_hash :
             associated_data.authorized_logical_pipeline_policies_hashes()) {
          authorized_logical_pipeline_policies_hashes_.insert(policy_hash);
        }
        FCP_RETURN_IF_ERROR(SetActiveKeyIds(
            {protected_response.decryption_keys().begin(),
             protected_response.decryption_keys().end()},
            {associated_data.omitted_decryption_key_ids().begin(),
             associated_data.omitted_decryption_key_ids().end()}));

        FCP_RETURN_IF_ERROR(
            StreamInitializeTransform(initialize_request.configuration(),
                                      associated_data.config_constraints()));
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

  {
    absl::MutexLock l(mutex_);
    if (decryptor_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "StreamInitialize can only be called once.");
    }
    decryptor_.emplace(std::vector<absl::string_view>(
        protected_response.decryption_keys().begin(),
        protected_response.decryption_keys().end()));
    session_tracker_.emplace(max_num_sessions);
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
  Decryptor* decryptor;
  {
    absl::MutexLock l(mutex_);
    if (decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }

    // Since decryptor_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    decryptor = &*decryptor_;
  }

  absl::StatusOr<SessionRequest> session_request = stream->Read();
  if (!session_request.ok()) {
    return session_request.status();
  }

  // Create the session.
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<confidential_federated_compute::Session> session,
      CreateSession());
  // This provides the context for encryption and writing of ReadResponse
  // messages.
  SessionContextImpl context(stream, kms_encryptor_);

  // If the first request is ConfigureRequest, it is used to configure the
  // session; otherwise a default ConfigureRequest is used to configure the
  // session.
  ConfigureRequest configure_request;
  if (session_request->has_configure()) {
    configure_request = std::move(*session_request->mutable_configure());
  }

  // ConfigureResponse is used further below in the loop to write the response.
  // The response is wrapped in std::optional to verify that only the first
  // ConfigureRequest received before other requests is valid.
  FCP_ASSIGN_OR_RETURN(
      std::optional<ConfigureResponse> configure_response,
      session->Configure(std::move(configure_request), context));

  // Initialize result_blob_metadata with unencrypted metadata since
  // EarliestExpirationTimeMetadata expects inputs to have either unencrypted or
  // hpke_plus_aead_data.
  BlobMetadata result_blob_metadata;
  result_blob_metadata.mutable_unencrypted();

  do {
    switch (session_request->kind_case()) {
      case SessionRequest::kConfigure: {
        if (!configure_response.has_value()) {
          // configure_response would have the value only if it was initialized
          // before the loop in the first request. This will fail on a
          // subsequent ConfigureRequest or any out of order ConfigureRequest.
          return absl::FailedPreconditionError(
              "Unexpected out of sequence ConfigureRequest");
        }
        SessionResponse session_response;
        *session_response.mutable_configure() =
            std::move(configure_response).value();
        context.ApplyCountersToResponse(&session_response);
        FCP_RETURN_IF_ERROR(stream->Write(session_response));
        break;
      }

      case SessionRequest::kWrite: {
        FCP_ASSIGN_OR_RETURN(
            WriteFinishedResponse write_response,
            HandleWrite(session.get(),
                        std::move(*session_request->mutable_write()), decryptor,
                        context));
        SessionResponse session_response;
        *session_response.mutable_write() = std::move(write_response);
        context.ApplyCountersToResponse(&session_response);
        FCP_RETURN_IF_ERROR(stream->Write(session_response));
        break;
      }

      case SessionRequest::kCommit: {
        FCP_ASSIGN_OR_RETURN(
            CommitResponse commit_response,
            session->Commit(session_request->commit(), context));
        SessionResponse session_response;
        *session_response.mutable_commit() = std::move(commit_response);
        context.ApplyCountersToResponse(&session_response);
        FCP_RETURN_IF_ERROR(stream->Write(session_response));
        break;
      }

      case SessionRequest::kFinalize: {
        FCP_ASSIGN_OR_RETURN(FinalizeResponse finalize_response,
                             session->Finalize(session_request->finalize(),
                                               result_blob_metadata, context));
        SessionResponse session_response;
        *session_response.mutable_finalize() = std::move(finalize_response);
        context.ApplyCountersToResponse(&session_response);
        FCP_RETURN_IF_ERROR(stream->Write(session_response));
        return absl::OkStatus();
      }

      default:
        return absl::FailedPreconditionError(
            absl::StrCat("Session received unexpected request of type: ",
                         session_request->kind_case()));
    }

    // Read the next request.
  } while ((session_request = stream->Read()).ok());

  return session_request.status();
}

grpc::Status ConfidentialTransformBase::StreamInitialize(
    ServerContext* context, grpc::ServerReader<StreamInitializeRequest>* reader,
    InitializeResponse* response) {
  return ToGrpcStatus(StreamInitializeInternal(reader, response));
}

grpc::Status ConfidentialTransformBase::Session(
    ServerContext* context,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  SessionTracker* session_tracker;
  {
    absl::MutexLock l(mutex_);
    if (session_tracker_ == std::nullopt) {
      return ToGrpcStatus(absl::FailedPreconditionError(
          "StreamInitialize must be called before Session."));
    }

    // Since session_tracker_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store
    // a local pointer to it and access the object without a lock after we
    // check under the mutex that a value has been set for the std::optional
    // wrapper.
    session_tracker = &*session_tracker_;
  }
  if (absl::Status session_status = session_tracker->AddSession();
      !session_status.ok()) {
    return ToGrpcStatus(session_status);
  }
  SessionStream session_stream(stream);
  auto status = SessionImpl(&session_stream);
  if (!status.ok()) {
    LOG(ERROR) << "SessionImpl failed: " << status;
  }

  absl::Status remove_session = session_tracker->RemoveSession();
  if (!remove_session.ok()) {
    return ToGrpcStatus(remove_session);
  }
  return ToGrpcStatus(status);
}

absl::StatusOr<Decryptor*> ConfidentialTransformBase::GetDecryptor() {
  absl::MutexLock l(mutex_);
  if (decryptor_ == std::nullopt) {
    return absl::FailedPreconditionError(
        "Initialize must be called before GetDecryptor.");
  }

  // Since decryptor_ is set once in Initialize and never modified,
  // and the underlying object is threadsafe, it is safe to store a local
  // pointer to it and access the object without a lock after we check under
  // the mutex that values have been set for the std::optional wrappers.
  return &*decryptor_;
}

}  // namespace confidential_federated_compute
