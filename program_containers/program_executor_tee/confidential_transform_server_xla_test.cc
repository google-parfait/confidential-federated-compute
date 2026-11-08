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

#include "program_executor_tee/confidential_transform_server_xla.h"

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "program_executor_tee/testing_base.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

// Register the global python environment.
::testing::Environment* const python_env =
    ::testing::AddGlobalTestEnvironment(new PythonEnvironment());

TYPED_TEST_SUITE(ProgramExecutorTeeSessionTest,
                 ::testing::Types<XLAProgramExecutorTeeConfidentialTransform>);

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithJAXComputation) {
  this->CreateSession(R"(
from federated_language_jax.computation import jax_computation
import tensorflow_federated as tff
import federated_language

def trusted_program(external_handle):

  @jax_computation.jax_computation
  def comp():
    return 10

  result = comp()
  result_val, _ = tff.framework.serialize_value(result, federated_language.framework.infer_type(result))
  external_handle.release_unencrypted(result_val.SerializeToString(), b"result")
  )");

  ::fcp::confidentialcompute::SessionRequest session_request;
  ::fcp::confidentialcompute::SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data.size(), 1);
  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(released_data["result"]);
  ASSERT_EQ(released_value.array().int32_list().value().size(), 1);
  ASSERT_EQ(released_value.array().int32_list().value().at(0), 10);

  auto released_state_changes =
      this->fake_data_read_write_service_.GetReleasedStateChanges();
  // There is no initial state.
  ASSERT_FALSE(released_state_changes["result"].first.value().has_value());
  // The first release operation triggers a state change that should decrease
  // the number of remaining runs and increment the counter.
  BudgetState expected_first_release_budget;
  expected_first_release_budget.set_counter(1);
  ASSERT_EQ(released_state_changes["result"].second.value(),
            expected_first_release_budget.SerializeAsString());

  ASSERT_TRUE(session_response.has_finalize());
}

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithJAXPrivacy) {
  this->CreateSession(R"(
import numpy as np
import jax_privacy
import tensorflow_federated as tff
import federated_language

def trusted_program(external_handle):
    fn = lambda param, data: 0.5 * np.mean((data - param) ** 2)
    grad_fn = jax_privacy.clipped_grad(fn, l2_clip_norm=np.inf)
    grad = grad_fn(3.0, np.array([0, 7, -2]))
    grad = grad.tolist()
    grad_val, _ = tff.framework.serialize_value(grad, federated_language.framework.infer_type(grad))
    external_handle.release_unencrypted(grad_val.SerializeToString(), b"result")
  )");

  ::fcp::confidentialcompute::SessionRequest session_request;
  ::fcp::confidentialcompute::SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data.size(), 1);
  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(released_data["result"]);
  ASSERT_EQ(released_value.array().float32_list().value().size(), 1);
  ASSERT_EQ(released_value.array().float32_list().value().at(0), 4.0);

  auto released_state_changes =
      this->fake_data_read_write_service_.GetReleasedStateChanges();
  // There is no initial state.
  ASSERT_FALSE(released_state_changes["result"].first.value().has_value());
  // The first release operation triggers a state change that should decrease
  // the number of remaining runs and increment the counter.
  BudgetState expected_first_release_budget;
  expected_first_release_budget.set_counter(1);
  ASSERT_EQ(released_state_changes["result"].second.value(),
            expected_first_release_budget.SerializeAsString());

  ASSERT_TRUE(session_response.has_finalize());
}

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithJAXPrivacyExecutionPlan) {
  this->CreateSession(R"(
import math
import federated_language
from federated_language_jax.computation import jax_computation
import jax.numpy as jnp
import jax_privacy
import numpy as np
import tensorflow_federated as tff

def trusted_program(external_handle):
  plan = jax_privacy.execution_plan.BandMFConfig.default(
      num_bands=2,
      iterations=4,
      expected_participations=1,
      strategy_optimization_steps=1,
  ).calibrate(epsilon=1.0, delta=1e-6).make()

  # The clipped gradient sum is computed by the XLA runtime.
  @jax_computation.jax_computation(
      np.float32, federated_language.TensorType(np.float32, (3,))
  )
  def clipped_grad_sum(param, data):
    loss_fn = lambda p, x: 0.5 * jnp.mean((x - p) ** 2)
    return plan.clipped_grad(loss_fn)(param, data)

  # Per-example gradients are [3, -4, 5], which are clipped to [1, -1, 1].
  grad = clipped_grad_sum(3.0, np.array([0, 7, -2], dtype=np.float32))

  noise_state = plan.noise_addition_transform.init(grad)
  noisy_grad, _ = plan.noise_addition_transform.update(grad, noise_state)
  noisy_grad = float(noisy_grad)
  assert math.isfinite(noisy_grad)

  grad = float(grad)
  grad_val, _ = tff.framework.serialize_value(grad, federated_language.framework.infer_type(grad))
  external_handle.release_unencrypted(grad_val.SerializeToString(), b"clipped_grad")
  noisy_grad_val, _ = tff.framework.serialize_value(noisy_grad, federated_language.framework.infer_type(noisy_grad))
  external_handle.release_unencrypted(noisy_grad_val.SerializeToString(), b"noisy_grad")
  )");

  ::fcp::confidentialcompute::SessionRequest session_request;
  ::fcp::confidentialcompute::SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data.size(), 2);
  tensorflow_federated::v0::Value clipped_grad_value;
  clipped_grad_value.ParseFromString(released_data["clipped_grad"]);
  ASSERT_EQ(clipped_grad_value.array().float32_list().value().size(), 1);
  EXPECT_NEAR(clipped_grad_value.array().float32_list().value().at(0), 1.0,
              1e-5);
  tensorflow_federated::v0::Value noisy_grad_value;
  noisy_grad_value.ParseFromString(released_data["noisy_grad"]);
  ASSERT_EQ(noisy_grad_value.array().float32_list().value().size(), 1);

  ASSERT_TRUE(session_response.has_finalize());
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee
