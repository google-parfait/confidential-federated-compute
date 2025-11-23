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

class ProgramExecutorTeeConfidentialTransformXlaSessionTest
    : public ProgramExecutorTeeSessionTest<
          XLAProgramExecutorTeeConfidentialTransform> {};

TEST_P(ProgramExecutorTeeConfidentialTransformXlaSessionTest,
       ProgramWithJAXComputation) {
  this->CreateSession(R"(
from federated_language_jax.computation import jax_computation
import tensorflow_federated as tff
import federated_language

def trusted_program(input_provider, external_service_handle):

  @jax_computation.jax_computation
  def comp():
    return 10

  result = comp()
  result_val, _ = tff.framework.serialize_value(result, federated_language.framework.infer_type(result))
  external_service_handle.release_unencrypted(result_val.SerializeToString(), b"result")
  )",
                      UseKms());

  ::fcp::confidentialcompute::SessionRequest session_request;
  ::fcp::confidentialcompute::SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto expected_request = fcp::confidentialcompute::outgoing::WriteRequest();
  expected_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("result");
  expected_request.set_commit(true);

  auto write_call_args = this->fake_data_read_write_service_.GetWriteCallArgs();
  ASSERT_EQ(write_call_args.size(), 1);
  ASSERT_EQ(write_call_args[0].size(), 1);

  auto write_request = write_call_args[0][0];
  ASSERT_EQ(write_request.first_request_metadata().unencrypted().blob_id(),
            "result");
  ASSERT_TRUE(write_request.commit());

  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(write_request.data());
  ASSERT_EQ(released_value.array().int32_list().value().size(), 1);
  ASSERT_EQ(released_value.array().int32_list().value().at(0), 10);

  ASSERT_TRUE(session_response.has_finalize());
}

TEST_P(ProgramExecutorTeeConfidentialTransformXlaSessionTest,
       ProgramWithJAXPrivacy) {
  this->CreateSession(R"(
import numpy as np
import jax_privacy
import tensorflow_federated as tff
import federated_language

def trusted_program(input_provider, external_service_handle):
    fn = lambda param, data: 0.5 * np.mean((data - param) ** 2)
    grad_fn = jax_privacy.clipped_grad(fn, l2_clip_norm=np.inf)
    grad = grad_fn(3.0, np.array([0, 7, -2]))
    grad = grad.tolist()
    grad_val, _ = tff.framework.serialize_value(grad, federated_language.framework.infer_type(grad))
    external_service_handle.release_unencrypted(grad_val.SerializeToString(), b"result")
  )",
                      UseKms());

  ::fcp::confidentialcompute::SessionRequest session_request;
  ::fcp::confidentialcompute::SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto expected_request = fcp::confidentialcompute::outgoing::WriteRequest();
  expected_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("result");
  expected_request.set_commit(true);

  auto write_call_args = this->fake_data_read_write_service_.GetWriteCallArgs();
  ASSERT_EQ(write_call_args.size(), 1);
  ASSERT_EQ(write_call_args[0].size(), 1);

  auto write_request = write_call_args[0][0];
  ASSERT_EQ(write_request.first_request_metadata().unencrypted().blob_id(),
            "result");
  ASSERT_TRUE(write_request.commit());

  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(write_request.data());
  ASSERT_EQ(released_value.array().float32_list().value().size(), 1);
  ASSERT_EQ(released_value.array().float32_list().value().at(0), 4.0);

  ASSERT_TRUE(session_response.has_finalize());
}

INSTANTIATE_TEST_SUITE_P(
    KmsParam, ProgramExecutorTeeConfidentialTransformXlaSessionTest,
    ::testing::Bool(),  // Generates {false, true}
    ProgramExecutorTeeSessionTest<
        ProgramExecutorTeeConfidentialTransform>::TestNameSuffix);

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee
