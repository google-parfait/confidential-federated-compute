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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_POBJECT_MAP_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_POBJECT_MAP_FN_H_

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
// and process them together in a single Map() call during Commit.
//
// This is the container-side counterpart of the Flume PObjectMapFn concept:
// instead of processing each input independently (like DoFn), PObjectMapFn
// buffers all inputs and processes them as a batch.
//
// Usage: subclass PObjectMapFn and implement Map().
//
// Memory note: all unencrypted Write() data is buffered in memory as
// Session::KV objects until Commit(). Subclasses processing large
// datasets should be aware of this memory profile.
class PObjectMapFn : public Fn {
 public:
  // Called once with ALL accumulated inputs from Write() calls and the
  // commit configuration. Implementations should process the inputs
  // and emit output(s) via context.Emit*().
  //
  // Returns an error status if an error occurred and the Fn should be
  // aborted. Metrics about ignorable errors can be recorded using the
  // Counters returned by Context::GetCounters.
  virtual absl::Status Map(google::protobuf::Any config,
                           std::vector<Session::KV> accumulated_inputs,
                           Context& context) = 0;

  // final: accumulates unencrypted_data into internal buffer, preserving
  // key and blob_id from the WriteRequest.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // final: calls Map() with all accumulated inputs. The buffer is moved
  // out and cleared before calling Map() for defensive cleanup.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override final;

 private:
  std::vector<Session::KV> accumulated_inputs_;
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_POBJECT_MAP_FN_H_
