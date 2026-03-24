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

// This file contains functions and classes for managing aggregation sessions
// of a ConfidentialTransform service.

#include "containers/session_stream.h"

#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute {

using ::fcp::confidentialcompute::MergeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteRequest;

// This struct is used to combine chunks in WriteRequest and MergeRequest.
struct ChunkedState {
  // The request from the first chunk containing metadata and configuration.
  // The data field of the request, which is a Cord, is used to combine chunks
  // together.
  std::variant<std::monostate, WriteRequest, MergeRequest> first_request;

  bool has_value() const { return first_request.index() != 0; }

  absl::Status FailedContinuationError() const {
    return absl::FailedPreconditionError(
        "Session expected a continuation of chunked blob "
        "but received another request");
  }

  absl::Status CheckContinuation() const {
    return has_value() ? FailedContinuationError() : absl::OkStatus();
  }

  template <typename T>
  absl::StatusOr<bool> Append(T* request) {
    bool is_commit = request->commit();
    if (has_value()) {
      // This is a continuation of chunked blob. Fail the request
      // if the stored first message is incompatible or if the appended
      // messages has metadata, indicating that it contains a new blob.
      T* compatible_first_request = std::get_if<T>(&first_request);
      if (!compatible_first_request || request->has_first_request_metadata()) {
        return FailedContinuationError();
      }

      absl::Cord data = compatible_first_request->data();
      data.Append(request->data());
      compatible_first_request->set_data(data);
    } else {
      // This is a new blob i.e. the first chunk of a multi-chunk blob or a
      // small blob consisting of a single chunk of data.
      first_request = std::move(*request);
    }
    if (is_commit) {
      *request = std::get<T>(std::move(first_request));
      request->set_commit(true);
    }
    return is_commit;
  }
};

// The implementation of Read combines consecutive WriteRequest and
// MergeRequest if they contain chunked blobs, but otherwise returns after
// reading a single request.
absl::StatusOr<SessionRequest> SessionStream::Read() {
  SessionRequest request;
  ChunkedState chunked_state;
  while (stream_->Read(&request)) {
    switch (request.kind_case()) {
      case SessionRequest::kWrite: {
        FCP_ASSIGN_OR_RETURN(bool is_commit,
                             chunked_state.Append(request.mutable_write()));
        if (is_commit) {
          // All chunks received.
          return request;
        }
        break;
      }
      case SessionRequest::kMerge: {
        FCP_ASSIGN_OR_RETURN(bool is_commit,
                             chunked_state.Append(request.mutable_merge()));
        if (is_commit) {
          // All chunks received.
          return request;
        }
        break;
      }
      case SessionRequest::kConfigure:
        // Overwrite the chunk size.
        FCP_RETURN_IF_ERROR(chunked_state.CheckContinuation());
        chunk_size_ = request.configure().chunk_size();
        if (chunk_size_ == 0) {
          return absl::InvalidArgumentError("Invalid chunk size: 0");
        }
        return request;

      default:
        FCP_RETURN_IF_ERROR(chunked_state.CheckContinuation());
        return request;
    }
  }
  return absl::AbortedError("Session failed to read request message");
}

absl::Status SessionStream::WriteImpl(const SessionResponse& response) {
  if (!stream_->Write(response)) {
    return absl::AbortedError("Session failed to write response message");
  }
  return absl::OkStatus();
}

// The implementation of Write splits ReadResponse into multiple outgoing
// ReadResponse messages containing chunks of blob if the blob size exceeds
// `chunk_size_'.
absl::Status SessionStream::Write(SessionResponse response) {
  if (response.has_read()) {
    ReadResponse* mutable_read = response.mutable_read();
    mutable_read->set_finish_read(mutable_read->data().size() <= chunk_size_);
    if (!mutable_read->finish_read()) {
      absl::Cord data = mutable_read->data();
      do {
        mutable_read->set_data(data.Subcord(0, chunk_size_));
        data.RemovePrefix(chunk_size_);
        FCP_RETURN_IF_ERROR(WriteImpl(response));
        mutable_read->clear_first_response_configuration();
        mutable_read->clear_first_response_metadata();
      } while (data.size() > chunk_size_);

      mutable_read->set_data(std::move(data));
      mutable_read->set_finish_read(true);
    }
  }

  // If the response is ReadResponse, write final chunk of data (or the only
  // chunk if the data was smaller or equal than the chunk size); otherwise
  // write the original response.
  return WriteImpl(response);
}

}  // namespace confidential_federated_compute