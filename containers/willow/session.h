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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_SESSION_H_

#include <vector>

#include "absl/status/statusor.h"
#include "containers/blob_metadata.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::willow {

// Willow implementation of Session interface.
class Session final : public confidential_federated_compute::Session {
 public:
  Session() = default;

  // Session configuration.
  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest request,
      Context& context) override;

  // Receives a single input.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest request,
      std::string unencrypted_data, Context& context) override;

  // Commits the batch of inputs.
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest request,
      Context& context) override;

  // Runs session finalization logic and completes the session.
  // After finalization, the session state is no longer mutable.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override;

 private:
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> AddInput(
      std::string input);
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Merge(
      std::string input);
  absl::StatusOr<std::string> Compact();
  absl::StatusOr<std::string> Finalize();
};

}  // namespace confidential_federated_compute::willow

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_WILLOW_SESSION_H_
