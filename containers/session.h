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

// This file contains functions and classes for managing aggregation sessions
// of a ConfidentialTransform service.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute {

// Class used to track the number of active sessions in a container.
//
// This class is threadsafe.
class SessionTracker {
 public:
  SessionTracker(int max_num_sessions)
      : available_sessions_(max_num_sessions),
        max_num_sessions_(max_num_sessions) {};

  // Tries to add a session and returns the amount of memory in bytes that the
  // session is allowed. Returns 0 if there is no available memory.
  absl::Status AddSession();

  // Tries to remove a session and returns an error if unable to do so.
  absl::Status RemoveSession();

 private:
  absl::Mutex mutex_;
  int available_sessions_ ABSL_GUARDED_BY(mutex_) = 0;
  int max_num_sessions_;
};

// Creates WriteFinishedResponse.
fcp::confidentialcompute::WriteFinishedResponse ToWriteFinishedResponse(
    absl::Status status, long committed_size_bytes = 0);

// Create a SessionResponse with a WriteFinishedResponse.
fcp::confidentialcompute::SessionResponse ToSessionWriteFinishedResponse(
    absl::Status status, long committed_size_bytes = 0);

// Creates CommitResponse.
fcp::confidentialcompute::CommitResponse ToCommitResponse(
    absl::Status status, int num_inputs_committed = 0,
    std::vector<absl::Status> ignored_errors = {});

// Interface for interacting with a session in a container. Implementations
// may not be threadsafe.
class Session {
 public:
  virtual ~Session() = default;

  // A struct that describes an input or output key/value pair with an
  // optional metadata
  struct KV {
    // Optional key associated with the data.
    // If not specified the key has the default value.
    google::protobuf::Any key;
    // Plaintext (unencrypted) data
    std::string data;
    // Blob ID associated with the data, if available; otherwise empty.
    std::string blob_id;

    // Implicit constructor that constructs KV from data.
    // This allows passing a string or literal in place of KV, for example
    // EmitUnencrypted("plaintext")
    template <typename T>
    KV(T&& data) : KV(google::protobuf::Any(), std::forward<T>(data)) {}

    // Explicit constructor
    KV(google::protobuf::Any key, std::string data,
       std::string blob_id = RandomBlobId())
        : key(std::move(key)),
          data(std::move(data)),
          blob_id(std::move(blob_id)) {}

    KV() = default;
    KV(const KV&) = default;
    KV(KV&&) = default;
    KV& operator=(const KV&) = default;
    KV& operator=(KV&&) = default;
  };

  // Context interface that provides ability to emit an an arbitrary number of
  // ReadResponse messages to the session stream.
  class Context {
   public:
    virtual ~Context() = default;

    // Emits a single ReadResponse message to the session stream. If necessary
    // the message may be chunked.
    // This is the raw level method that takes the prepared ReadResponse and
    // emits it as is without any further processing other than chunking.
    // If the data needs to be encrypted, that has to be done in advance.
    virtual bool Emit(fcp::confidentialcompute::ReadResponse read_response) = 0;

    // Emits the provided key/value as plaintext, without encryption.
    virtual bool EmitUnencrypted(KV kv) = 0;

    // Encrypts and emits the provided key/value using the specified
    // reencryption key. This methods is appropriate only for temporary
    // encryption, for blobs that will be consumed by other parts of the
    // pipeline. Use `EmitReleasable` for emitting the final releasable result.
    virtual bool EmitEncrypted(int reencryption_key_index, KV kv) = 0;

    // Encrypts and emits the releasable result using the provided key/value and
    // the specified reencryption key. Also, saves the `release_token` in the
    // passed in field. The `release_token` should be passed back to the host
    // via FinalizeResponse.
    virtual bool EmitReleasable(int reencryption_key_index, KV kv,
                                std::optional<absl::string_view> src_state,
                                absl::string_view dst_state,
                                std::string& release_token) = 0;

    // Emits an error to the session stream but does not close the stream. This
    // error can be logged and used to increment Flume counters on the untrusted
    // side.
    virtual bool EmitError(absl::Status status) = 0;
  };

  // Initialize the session with the given configuration.
  virtual absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest request, Context& context) = 0;

  // Process a write request, optionally caching it to later incorporate into
  // the session upon receiving commit request.
  virtual absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest request,
      std::string unencrypted_data, Context& context) = 0;

  // Incorporate any cached write requests into the session.
  virtual absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest request, Context& context) = 0;

  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  virtual absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) = 0;
};

}  // namespace confidential_federated_compute
#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_
