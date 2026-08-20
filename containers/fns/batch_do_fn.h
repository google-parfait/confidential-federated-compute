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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_BATCH_DO_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_BATCH_DO_FN_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// Base class for Fn implementations that accumulate all Write() inputs
// and process them together in a single Do() call during Commit.
//
// Instead of processing each input independently (like DoFn), BatchDoFn
// buffers all inputs and processes them as a batch.
//
// Typical lifecycle:
//   1. Configure → calls InitializeReplica (setup)
//   2. Write (1..N) → buffers unencrypted inputs into memory.
//   3. Commit → calls Do() with all accumulated inputs.
//        Do() receives a DoContext which supports emitting intermediate
//        (non-releasable) blobs only.
//   4. Finalize → calls FinalizeReplica() to emit final output.
//        FinalizeReplica() receives an FnContext which supports
//        emitting releasable blobs via FnContext::EmitReleasable.
//        The release token is automatically captured and returned in the
//        FinalizeResponse.
//
// Usage: subclass BatchDoFn and implement Do(), and optionally override
// FinalizeReplica() to emit releasable results.
//
// Memory note: all unencrypted Write() data is buffered in memory as
// Session::KV objects until Commit(). Subclasses processing large
// datasets should be aware of this memory profile.
class BatchDoFn : public Fn {
 public:
  // A context for the Do() method that intentionally hides EmitReleasable
  // and GetReleaseToken. This ensures that releasable blobs can only be
  // emitted during FinalizeReplica(), not during Do().
  //
  // All other FnContext methods (Emit, EmitUnencrypted, EmitEncrypted,
  // metadata, counters) remain accessible.
  class DoContext : public FnContext {
   public:
    DoContext(Context& session_context,
              fcp::confidentialcompute::AssociatedMetadata metadata)
        : FnContext(session_context, std::move(metadata)) {}

   private:
    // Hide EmitReleasable and GetReleaseToken from Do() callers.
    using FnContext::EmitReleasable;
    using FnContext::GetReleaseToken;
  };

  // Called once with ALL accumulated inputs from Write() calls and the
  // commit configuration. Implementations should process the inputs
  // and emit output(s) via context.Emit*().
  //
  // Each input KV carries its associated_metadata. The DoContext starts with
  // empty metadata; use PackMetadata() or set associated_metadata on output KVs
  // directly.
  //
  // Returns an error status if an error occurred and the Fn should be
  // aborted. Metrics about ignorable errors can be recorded using the
  // Counters via DoContext::IncrementCounter or DoContext::IncrementCounterBy.
  virtual absl::Status Do(google::protobuf::Any config,
                          std::vector<Session::KV> accumulated_inputs,
                          DoContext& context) = 0;

  // final: accumulates unencrypted_data into internal buffer, preserving
  // key and blob_id from the WriteRequest.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // final: calls Do() with all accumulated inputs. The buffer is moved
  // out and cleared before calling Do() for defensive cleanup.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override final;

 private:
  std::vector<Session::KV> accumulated_inputs_;
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_BATCH_DO_FN_H_
