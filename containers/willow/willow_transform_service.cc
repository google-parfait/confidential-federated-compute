// Copyright 2026 Google LLC.
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

#include "willow_transform_service.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
// #include "containers/blob_metadata.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"
#include "willow_session.h"

namespace confidential_federated_compute::willow {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::grpc::ServerContext;

grpc::Status WillowTransformService::StreamInitialize(
    ServerContext* context, grpc::ServerReader<StreamInitializeRequest>* reader,
    InitializeResponse* response) {
  return ToGrpcStatus(StreamInitializeImpl(reader, response));
}

grpc::Status WillowTransformService::Session(ServerContext* context,
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

absl::Status WillowTransformService::StreamInitializeImpl(
    grpc::ServerReader<fcp::confidentialcompute::StreamInitializeRequest>*
        reader,
    fcp::confidentialcompute::InitializeResponse* response) {
  StreamInitializeRequest request;
  std::optional<google::protobuf::Any> configuration;
  uint32_t max_num_sessions;
  while (reader->Read(&request)) {
    switch (request.kind_case()) {
      case StreamInitializeRequest::kInitializeRequest: {
        // Only a single initialize request is supported.
        if (configuration.has_value()) {
          return absl::FailedPreconditionError(
              "StreamInitializeRequests: received more than one "
              "InitializeRequest.");
        }
        max_num_sessions = request.initialize_request().max_num_sessions();
        configuration.emplace(std::move(
            *request.mutable_initialize_request()->mutable_configuration()));
        break;
      }
      case StreamInitializeRequest::kWriteConfiguration: {
        return absl::FailedPreconditionError(
            "StreamInitializeRequest: WriteConfigurationRequest is not "
            "supported for WillowTransformService");
      }
      default:
        return absl::FailedPreconditionError(absl::StrCat(
            "StreamInitializeRequest: unexpected request of type: ",
            request.kind_case()));
    }
  }
  if (!configuration.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("StreamInitializeRequest: expected one InitializeRequest "
                     "but received zero."));
  }

  {
    absl::MutexLock l(&mutex_);
    if (IsInitialized()) {
      return absl::FailedPreconditionError(
          "StreamInitialize can only be called once.");
    }
    FCP_RETURN_IF_ERROR(InitializeTransform(configuration.value()));
    session_tracker_.emplace(max_num_sessions);
  }

  return absl::OkStatus();
}

absl::Status WillowTransformService::InitializeTransform(
    const google::protobuf::Any& configuration) {
  if (!configuration.UnpackTo(&aggregation_config_)) {
    return absl::InvalidArgumentError(
        "Failed to parse Willow aggregation configuration");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Session>>
WillowTransformService::CreateSession() {
  return std::make_unique<WillowSession>(aggregation_config_);
};

// Implementation of Session::Context that is passed to the
// session methods to emit resulting blobs. Encryption of
// outgoing blobs isn't supported in Willow Session.
class ContextImpl : public Session::Context {
 public:
  ContextImpl(SessionStream* stream) : stream_(stream) {}

  bool Emit(ReadResponse read_response) override {
    SessionResponse response;
    ReadResponse* mutable_read = response.mutable_read();
    *mutable_read = std::move(read_response);
    // TODO: implement chunking
    mutable_read->set_finish_read(true);
    return stream_->Write(response);
  }

  bool Emit(std::string data, google::protobuf::Any key,
            BlobMetadata metadata) {
    ReadResponse response;
    *response.mutable_data() = std::move(data);
    *response.mutable_first_response_configuration() = std::move(key);
    *response.mutable_first_response_metadata() = std::move(metadata);
    return Emit(std::move(response));
  }

  bool EmitUnencrypted(Session::KV kv) override {
    BlobMetadata metadata;
    metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
    metadata.set_total_size_bytes(kv.data.size());
    metadata.mutable_unencrypted()->set_blob_id(kv.blob_id);
    return Emit(std::move(kv.data), std::move(kv.key), std::move(metadata));
  }

  // Methods below aren't supported in Willow Session implementation.
  bool EmitEncrypted(int reencryption_key_index, Session::KV kv) override {
    return false;
  }
  bool EmitReleasable(int reencryption_key_index, Session::KV kv,
                      std::optional<absl::string_view> src_state,
                      absl::string_view dst_state,
                      std::string& release_token) override {
    return false;
  }
  bool EmitError(absl::Status status) override { return false; }

 private:
  SessionStream* stream_;
};

absl::Status WillowTransformService::SessionImpl(SessionStream* stream) {
  {
    absl::MutexLock l(&mutex_);
    if (!IsInitialized()) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }
  }

  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<confidential_federated_compute::Session> session,
      CreateSession());

  ContextImpl context(stream);

  // Base implementation of the SessionRequest handling.
  // TODO: refactor to share the blob chunking with
  // ConfidentialTransformServerBase
  SessionRequest request;
  while (stream->Read(&request)) {
    switch (request.kind_case()) {
      case SessionRequest::kConfigure: {
        // Ignore the session configuration for now and just generate the
        // successful response.
        SessionResponse response;
        response.mutable_configure();
        stream->Write(response);
        break;
      }

      case SessionRequest::kWrite: {
        std::string data = std::move(*request.mutable_write()->mutable_data());
        FCP_ASSIGN_OR_RETURN(
            WriteFinishedResponse write_response,
            session->Write(request.write(), std::move(data), context));
        SessionResponse response;
        *response.mutable_write() = std::move(write_response);
        stream->Write(response);
        break;
      }

      case SessionRequest::kCommit: {
        FCP_ASSIGN_OR_RETURN(CommitResponse commit_response,
                             session->Commit(request.commit(), context));
        SessionResponse response;
        *response.mutable_commit() = std::move(commit_response);
        stream->Write(response);
        break;
      }

      case SessionRequest::kFinalize: {
        // TODO: refactor the Session::Finalize so that it no longer
        // requires the result_blob_metadata argument.
        BlobMetadata result_blob_metadata;
        result_blob_metadata.mutable_unencrypted();
        FCP_ASSIGN_OR_RETURN(FinalizeResponse finalize_response,
                             session->Finalize(request.finalize(),
                                               result_blob_metadata, context));
        SessionResponse response;
        *response.mutable_finalize() = std::move(finalize_response);
        stream->Write(response);
        return absl::OkStatus();
      }

      default:
        return absl::FailedPreconditionError(
            absl::StrCat("Session received an unexpected request of type: ",
                         request.kind_case()));
    }
  }

  return absl::AbortedError(
      "Session failed to read client write or finalize message.");
}

}  // namespace confidential_federated_compute::willow
