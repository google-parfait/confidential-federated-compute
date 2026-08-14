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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_DO_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_DO_FN_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// Session base class for DoFns.
//
// DoFn processes inputs one at a time via Do() (called from Write) and
// optionally produces final output via FinalizeReplica() (called from
// Finalize).
//
// Typical lifecycle:
//   1. Configure → calls InitializeReplica (setup)
//   2. Write (1..N) → each calls Do() to process a single input element.
//        Do() receives a DoContext which supports emitting intermediate
//        (non-releasable) blobs only.
//   3. Finalize → calls FinalizeReplica() to emit final output.
//        FinalizeReplica() receives an FnContext which supports
//        emitting releasable blobs via FnContext::EmitReleasable.
//        The release token is automatically captured and returned in the
//        FinalizeResponse.
class DoFn : public Fn {
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

  // Processes an input element. The input Value.data is unencrypted. Uses the
  // DoContext to emit zero or more non-releasable output elements.
  //
  // Returns an error status if an error occurred and the Fn should be aborted.
  // This is equivalent to calling AbortReplica in Flume. Metrics about
  // ignorable errors can be recorded using the Counters returned by
  // DoContext::GetCounters.
  virtual absl::Status Do(KV input, DoContext& context) = 0;

  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // A no-op by default.
  virtual absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override {
    return fcp::confidentialcompute::CommitResponse();
  }
};
}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_DO_FN_H_
