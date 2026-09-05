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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_TESTING_MOCK_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_TESTING_MOCK_FN_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// MockFn provides a gmock-based mock for all Fn virtual/override methods.
// Fn declares MockFn as a friend so it can inherit from Fn directly.
class MockFn : public Fn {
 public:
  MOCK_METHOD(absl::Status, InitializeReplica,
              (google::protobuf::Any config, ConfigureContext& context),
              (override));
  MOCK_METHOD(absl::Status, FinalizeReplica,
              (google::protobuf::Any config, FnContext& context), (override));
  MOCK_METHOD((absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>),
              Write,
              (fcp::confidentialcompute::WriteRequest write_request,
               std::string unencrypted_data, Context& context),
              (override));
  MOCK_METHOD((absl::StatusOr<fcp::confidentialcompute::CommitResponse>),
              Commit,
              (fcp::confidentialcompute::CommitRequest commit_request,
               Context& context),
              (override));
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_TESTING_MOCK_FN_H_
