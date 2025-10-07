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
#include "confidential_transform_server.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "program_executor_tee/testing_base.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::tensorflow::program_executor_tee {

namespace {

using ::confidential_federated_compute::program_executor_tee::
    ProgramExecutorTeeSessionTest;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;

TYPED_TEST_SUITE(
    ProgramExecutorTeeSessionTest,
    ::testing::Types<TensorflowProgramExecutorTeeConfidentialTransform>);

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithDataSource) {
  this->CreateSession(
      R"(
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np

async def trusted_program(input_provider, release_manager):
  model = input_provider.get_model('model1')
  def model_fn() -> tff.learning.models.VariableModel:
    return tff.learning.models.model_from_functional(model)

  learning_process = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=tff.learning.optimizers.build_sgdm(
          learning_rate=0.01
      ),
  )
  state = learning_process.initialize()
  await release_manager.release(state, "result")
  )",
      /*client_ids=*/{}, /*client_data_dir=*/"",
      /*file_id_to_filepath=*/
      {{"model1", "testdata/model1.zip"}});

  SessionRequest session_request;
  SessionResponse session_response;
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
  ASSERT_EQ(released_value.struct_().element().size(), 5);

  ASSERT_TRUE(session_response.has_finalize());
}

}  // namespace

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee
