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

#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

// Session base class for DoFns.
class DoFn : public Fn {
 public:
  // Processes an input element. The input Value.data is unencrypted. Uses the
  // Context to emit zero or more output elements.
  //
  // Returns an error status if an error occurred and the Fn should be aborted.
  // This is equivalent to calling AbortReplica in Flume. Ignorable errors
  // should be emitted using Context::EmitError.
  virtual absl::Status Do(KV input, Context& context) = 0;

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
