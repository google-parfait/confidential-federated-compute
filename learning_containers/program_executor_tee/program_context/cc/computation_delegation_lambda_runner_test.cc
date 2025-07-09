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

#include "learning_containers/program_executor_tee/program_context/cc/computation_delegation_lambda_runner.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mock_noise_client_session.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidentialcompute::TffSessionConfig;
using ::oak::session::v1::PlaintextMessage;
using ::tensorflow_federated::v0::Value;
using ::testing::_;
using ::testing::Return;

constexpr char kWorkerBns[] = "/bns/test/worker";
constexpr int kNumClients = 3;

Value GetTestLambdaFunction() {
  Value function;
  *function.mutable_computation()
       ->mutable_lambda()
       ->mutable_result()
       ->mutable_type() = PARSE_TEXT_PROTO(R"pb(tensor { dtype: DT_INT32 })pb");
  return function;
}

TEST(ComputationDelegationLambdaRunnerTest, ExecuteCompSucceeds) {
  auto mock_noise_client_session_ptr =
      std::make_unique<MockNoiseClientSession>();
  auto mock_noise_client_session = mock_noise_client_session_ptr.get();
  ComputationDelegationLambdaRunner runner(mock_noise_client_session);

  auto test_fn = GetTestLambdaFunction();
  auto test_arg = Value();

  TffSessionConfig tff_session_config;
  *tff_session_config.mutable_function() = test_fn;
  *tff_session_config.mutable_initial_arg() = test_arg;
  tff_session_config.set_num_clients(kNumClients);

  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(tff_session_config.SerializeAsString());
  // Return an empty Value as the result.
  Value result = Value();
  PlaintextMessage plaintext_response;
  plaintext_response.set_plaintext(result.SerializeAsString());
  EXPECT_CALL(*mock_noise_client_session,
              DelegateComputation(EqualsProto(plaintext_request)))
      .WillOnce(Return(plaintext_response));

  auto execute = runner.ExecuteComp(test_fn, test_arg, kNumClients);
  EXPECT_OK(execute);
  EXPECT_THAT(execute.value(), EqualsProto(result));
}

TEST(ComputationDelegationLambdaRunnerTest, ExecuteCompFailsAtExecution) {
  auto mock_noise_client_session_ptr =
      std::make_unique<MockNoiseClientSession>();
  auto mock_noise_client_session = mock_noise_client_session_ptr.get();
  ComputationDelegationLambdaRunner runner(mock_noise_client_session);

  auto test_fn = GetTestLambdaFunction();
  auto test_arg = Value();

  TffSessionConfig tff_session_config;
  *tff_session_config.mutable_function() = test_fn;
  *tff_session_config.mutable_initial_arg() = test_arg;
  tff_session_config.set_num_clients(kNumClients);

  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(tff_session_config.SerializeAsString());

  EXPECT_CALL(*mock_noise_client_session,
              DelegateComputation(EqualsProto(plaintext_request)))
      .WillOnce(Return(absl::InternalError("Failed to delegate computation.")));

  auto execute = runner.ExecuteComp(test_fn, test_arg, kNumClients);
  EXPECT_EQ(execute.status(),
            absl::InternalError("Failed to delegate computation."));
}

TEST(ComputationDelegationLambdaRunnerTest, ExecuteCompFailsAtParsing) {
  auto mock_noise_client_session_ptr =
      std::make_unique<MockNoiseClientSession>();
  auto mock_noise_client_session = mock_noise_client_session_ptr.get();
  ComputationDelegationLambdaRunner runner(mock_noise_client_session);

  auto test_fn = GetTestLambdaFunction();
  auto test_arg = Value();

  TffSessionConfig tff_session_config;
  *tff_session_config.mutable_function() = test_fn;
  *tff_session_config.mutable_initial_arg() = test_arg;
  tff_session_config.set_num_clients(kNumClients);

  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext(tff_session_config.SerializeAsString());

  PlaintextMessage plaintext_response;
  plaintext_response.set_plaintext("invalid value");
  EXPECT_CALL(*mock_noise_client_session,
              DelegateComputation(EqualsProto(plaintext_request)))
      .WillOnce(Return(plaintext_response));

  auto execute = runner.ExecuteComp(test_fn, test_arg, kNumClients);
  EXPECT_EQ(execute.status(),
            absl::InternalError("Failed to parse response as tff Value."));
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee