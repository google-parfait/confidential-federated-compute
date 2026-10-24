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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_GLOBAL_REDUCE_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_GLOBAL_REDUCE_FN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// Base class for Fns that reduce all inputs of a session to a single output,
// which is released along with a private state transition. Reduce() has a
// global view of the inputs: it's called exactly once, with all of them.
//
// Lifecycle:
//   1. Configure → calls InitializeReplica (setup).
//   2. Write (0..N) → rejects duplicate privacy IDs (see below) and buffers the
//        unencrypted inputs in memory.
//   3. Commit → records that all inputs have been written.
//   4. Finalize → calls Reduce() exactly once with all buffered inputs and
//        releases the output returned by Reduce() via
//        FnContext::EmitReleasable. The release token is automatically
//        captured and returned in the FinalizeResponse.
//
// Privacy IDs: if the inputs carry a PrivacyIdMetadata entry in their protected
// metadata, no two inputs may share a privacy ID, i.e. come from the same
// privacy unit, so each privacy unit contributes at most one input to the
// released output. Either all inputs must carry a PrivacyIdMetadata or none
// may. A Write() that violates this fails with INVALID_ARGUMENT, which aborts
// the session.
//
// Scale: all inputs are processed by a single session in a single TEE, and all
// of them are buffered in memory until Finalize. GlobalReduceFn is meant for
// computations that need all inputs at once and whose inputs fit in memory; it
// isn't a scalable aggregation primitive.
//
// Usage: subclass GlobalReduceFn and implement Reduce(), and optionally
// override InitializeReplica() and GetReencryptionKeyIndex().
class GlobalReduceFn : public Fn {
 public:
  // A context for the Reduce() method, which only allows recording counters.
  // Reduce() returns exactly one output, which GlobalReduceFn releases.
  class ReduceContext {
   public:
    // Increments the named counter by 1.
    void IncrementCounter(absl::string_view name) {
      context_.IncrementCounter(name);
    }
    // Increments the named counter by the given amount.
    void IncrementCounterBy(absl::string_view name, int64_t amount) {
      context_.IncrementCounterBy(name, amount);
    }

   private:
    friend class GlobalReduceFn;
    explicit ReduceContext(FnContext& context) : context_(context) {}

    FnContext& context_;
  };

  // A single element buffered from a Write() call.
  struct Element {
    Session::KV kv;
    fcp::confidentialcompute::AssociatedMetadata associated_metadata;
    fcp::confidentialcompute::ProtectedMetadata protected_metadata;
  };

  // The single output returned by Reduce(), which is released during Finalize.
  struct ReleasableOutput {
    // The output to release.
    Session::KV kv;
    // The private state (e.g. a serialized budget) that the release
    // transitions from, or std::nullopt if there is no prior state.
    std::optional<std::string> src_state;
    // The private state that the release transitions to.
    std::string dst_state;
  };

  // Called exactly once, during Finalize, with all inputs in the order in
  // which they were written. Returns the single output to release, along with
  // its private state transition, or an error status if the Fn should be
  // aborted. Metrics can be recorded using ReduceContext::IncrementCounter or
  // ReduceContext::IncrementCounterBy.
  virtual absl::StatusOr<ReleasableOutput> Reduce(std::vector<Element> inputs,
                                                  ReduceContext& context) = 0;

  // Returns the index of the reencryption key that is used to release the
  // output. Returns 0 by default.
  virtual int GetReencryptionKeyIndex() const { return 0; }

  // final: rejects duplicate privacy IDs (see above), then buffers
  // unencrypted_data, preserving key, blob_id and metadata from the
  // WriteRequest. Fails if Commit() has already been called.
  using Fn::Write;
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data,
      fcp::confidentialcompute::ProtectedMetadata protected_metadata,
      Context& context) override final;

  // final: records that all inputs have been written. Fails if called more
  // than once.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override final;

  // final: calls Reduce() with all buffered inputs and releases the output it
  // returns via EmitReleasable. Fails if Commit() hasn't been called, or if
  // called more than once.
  absl::Status FinalizeReplica(google::protobuf::Any config,
                               FnContext& context) override final;

 private:
  // Returns an error if `input` shares a privacy ID with a previously written
  // input, or if only some of the inputs have a privacy ID. Otherwise, records
  // the privacy ID of `input`, if any, so that later duplicates are detected.
  absl::Status CheckPrivacyId(const Element& input);

  std::vector<Element> inputs_;
  // Whether Commit() has been called.
  bool committed_ = false;
  // Whether FinalizeReplica() has been called.
  bool finalized_ = false;
  // Whether the inputs have a privacy ID. Determined by the first input.
  std::optional<bool> has_privacy_ids_;
  // The privacy IDs of the inputs written so far.
  absl::flat_hash_set<std::string> seen_privacy_ids_;
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_GLOBAL_REDUCE_FN_H_
