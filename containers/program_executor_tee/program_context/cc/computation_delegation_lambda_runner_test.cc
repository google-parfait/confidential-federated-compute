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

#include "containers/program_executor_tee/program_context/cc/computation_delegation_lambda_runner.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "fcp/protos/confidentialcompute/computation_delegation_mock.grpc.pb.h"
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
using ::fcp::confidentialcompute::outgoing::MockComputationDelegationStub;
using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::PlaintextMessage;
using ::tensorflow_federated::v0::Value;
using ::testing::_;
using ::testing::Return;

constexpr char kWorkerBns[] = "/bns/test/worker";
constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";
constexpr int kNumClients = 3;

SessionConfig* TestConfigAttestedNNClient() {
  auto verifier = bindings::new_fake_attestation_verifier(
      ffi_bindings::BytesView(kFakeEvent),
      ffi_bindings::BytesView(kFakePlatform));

  return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                              HandshakeType::kNoiseNN)
      .AddPeerVerifier(kFakeAttesterId, verifier)
      .Build();
}

Value GetTestLambdaFunction() {
  Value function;
  *function.mutable_computation()
       ->mutable_lambda()
       ->mutable_result()
       ->mutable_type() = PARSE_TEXT_PROTO(R"pb(tensor { dtype: DT_INT32 })pb");
  return function;
}

TEST(ComputationDelegationLambdaRunnerTest, CreateSucceeds) {
  auto mock_stub = new MockComputationDelegationStub();
  auto runner = ComputationDelegationLambdaRunner::Create(
      kWorkerBns, TestConfigAttestedNNClient(), mock_stub);
  EXPECT_OK(runner);
}

TEST(ComputationDelegationLambdaRunnerTest, CreateFailsWithEmptyWorkerBns) {
  auto mock_stub = new MockComputationDelegationStub();
  auto runner = ComputationDelegationLambdaRunner::Create(
      "", TestConfigAttestedNNClient(), mock_stub);
  EXPECT_EQ(runner.status(),
            absl::InvalidArgumentError("Worker bns is empty."));
}

TEST(ComputationDelegationLambdaRunnerTest, ExecuteCompSucceeds) {
  auto mock_noise_client_session_ptr =
      std::make_unique<MockNoiseClientSession>();
  auto mock_noise_client_session = mock_noise_client_session_ptr.get();
  ComputationDelegationLambdaRunner runner(
      std::move(mock_noise_client_session_ptr));

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
  ComputationDelegationLambdaRunner runner(
      std::move(mock_noise_client_session_ptr));

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
  ComputationDelegationLambdaRunner runner(
      std::move(mock_noise_client_session_ptr));

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