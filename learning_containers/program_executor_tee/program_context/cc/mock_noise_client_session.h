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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_MOCK_NOISE_CLIENT_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_MOCK_NOISE_CLIENT_SESSION_H_

#include "gmock/gmock.h"
#include "learning_containers/program_executor_tee/program_context/cc/noise_client_session.h"
#include "proto/session/session.pb.h"

namespace confidential_federated_compute::program_executor_tee {

class MockNoiseClientSession : public NoiseClientSessionInterface {
 public:
  MOCK_METHOD(absl::StatusOr<oak::session::v1::PlaintextMessage>,
              DelegateComputation,
              (const oak::session::v1::PlaintextMessage& plaintext_request),
              (override));
};
}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_MOCK_NOISE_CLIENT_SESSION_H_
