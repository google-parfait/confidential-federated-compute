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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_MAP_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_MAP_FN_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

// Session base class for MapFns.
class MapFn : public Fn {
 public:
  // A context for the Map() method that intentionally hides all Emit methods.
  // Map returns exactly one output value; the implementation automatically
  // handles emission of that value.
  //
  // Metadata methods and GetCounters remain accessible.
  class MapContext : public FnContext {
   public:
    MapContext(Context& session_context,
               fcp::confidentialcompute::AssociatedMetadata metadata)
        : FnContext(session_context, std::move(metadata)) {}

   private:
    // Hide all Emit methods — Map returns exactly one value,
    // Write() handles emission.
    using FnContext::Emit;
    using FnContext::EmitEncrypted;
    using FnContext::EmitReleasable;
    using FnContext::EmitUnencrypted;
  };

  // Processes an input element. The input KV.data is unencrypted. Returns a
  // KV containing the corresponding output element along with any
  // metadata.
  virtual absl::StatusOr<KV> Map(KV input, MapContext& context) = 0;

  // Controls how the output KV is emitted. Override to return a
  // reencryption key index for encrypted emission. Returns std::nullopt by
  // default, which emits the output unencrypted.
  virtual std::optional<int> GetReencryptionKeyIndex() const {
    return std::nullopt;
  }

  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // No-op for MapFn.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override final {
    return fcp::confidentialcompute::CommitResponse();
  }
};
}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_MAP_FN_H_
