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

#include "program_executor_tee/program_context/cc/computation_delegation_lambda_runner.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "program_executor_tee/program_context/cc/noise_client_session.h"
#include "proto/session/session.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidentialcompute::TffSessionConfig;
using ::oak::session::v1::PlaintextMessage;
using ::tensorflow_federated::v0::Value;

absl::StatusOr<Value> ComputationDelegationLambdaRunner::ExecuteComp(
    Value function, std::optional<Value> arg, int32_t num_clients) {
  absl::MutexLock lock(&mutex_);
  TffSessionConfig tff_session_config;
  *tff_session_config.mutable_function() = function;
  if (arg.has_value()) {
    *tff_session_config.mutable_initial_arg() = arg.value();
  }
  tff_session_config.set_num_clients(num_clients);
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(tff_session_config.SerializeAsString());
  FCP_ASSIGN_OR_RETURN(
      PlaintextMessage plaintext_response,
      noise_client_session_->DelegateComputation(plaintext_request));
  Value result;
  if (!result.ParseFromString(plaintext_response.plaintext())) {
    return absl::InternalError("Failed to parse response as tff Value.");
  }
  return result;
}

}  // namespace confidential_federated_compute::program_executor_tee
