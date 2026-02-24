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
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "program_executor_tee/testing_base.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::tensorflow::program_executor_tee {

namespace {

using ::confidential_federated_compute::program_executor_tee::BudgetState;
using ::confidential_federated_compute::program_executor_tee::
    BuildClientCheckpointFromInts;
using ::confidential_federated_compute::program_executor_tee::kMaxNumRuns;
using ::confidential_federated_compute::program_executor_tee::
    ProgramExecutorTeeConfidentialTransform;
using ::confidential_federated_compute::program_executor_tee::
    ProgramExecutorTeeSessionTest;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;

// Register the global python environment.
::testing::Environment* const python_env = ::testing::AddGlobalTestEnvironment(
    new ::confidential_federated_compute::program_executor_tee::
        PythonEnvironment());

TYPED_TEST_SUITE(
    ProgramExecutorTeeSessionTest,
    ::testing::Types<TensorflowProgramExecutorTeeConfidentialTransform>);

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithDataSource) {
  std::vector<std::string> client_ids = {"client1", "client2", "client3",
                                         "client4"};
  std::string client_data_dir = "data_dir";
  std::string tensor_name = "output_tensor_name";
  for (int i = 0; i < client_ids.size(); i++) {
    std::string data = BuildClientCheckpointFromInts(
        {1 + i * 3, 2 + i * 3, 3 + i * 3}, tensor_name);
    CHECK_OK(this->fake_data_read_write_service_.StoreEncryptedMessageForKms(
        client_data_dir + "/" + client_ids[i], data));
  }

  this->CreateSession(R"(
import collections
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from google.protobuf import any_pb2
from fcp.confidentialcompute.python import min_sep_data_source

def trusted_program(input_provider, external_service_handle):

  data_source = min_sep_data_source.MinSepDataSource(
      min_sep=2,
      input_provider=input_provider,
      computation_type=computation_pb2.Type(
          tensor=computation_pb2.TensorType(
              dtype=data_type_pb2.DataType.DT_INT32,
              dims=[3],
          )
      ),
  )
  data_source_iterator = data_source.iterator()

  client_data_type = federated_language.FederatedType(
      federated_language.TensorType(np.int32, [3]), federated_language.CLIENTS
  )

  server_data_type = federated_language.FederatedType(
      federated_language.StructType([
          ('sum', federated_language.TensorType(np.int32, [3])),
          ('client_count', federated_language.TensorType(np.int32, [])),
      ]),
      federated_language.SERVER,
  )

  @tff.tensorflow.computation
  def add(x, y):
    return x + y

  @federated_language.federated_computation(server_data_type, client_data_type)
  def my_comp(server_state, client_data):
    summed_client_data = federated_language.federated_sum(client_data)
    client_count = federated_language.federated_sum(
        federated_language.federated_value(1, federated_language.CLIENTS)
    )
    return tff.learning.templates.LearningProcessOutput(
        federated_language.federated_zip(
            collections.OrderedDict(
                sum=federated_language.federated_map(
                    add, (server_state.sum, summed_client_data)
                ),
                client_count=federated_language.federated_map(
                    add, (server_state.client_count, client_count)
                ),
            )
        ),
        client_count,
    )

  # Run four rounds, which will guarantee that each client is used exactly twice.
  server_state = {'sum': [0, 0, 0], 'client_count': 0}
  for _ in range(4):
    server_state, metrics = my_comp(server_state, data_source_iterator.select(2))

  sum_val, _ = tff.framework.serialize_value(
      server_state["sum"],
      federated_language.framework.infer_type(server_state["sum"]),
  )
  client_count_val, _ = tff.framework.serialize_value(
      server_state["client_count"],
      federated_language.framework.infer_type(server_state["client_count"]),
  )
  external_service_handle.release_unencrypted(
      sum_val.SerializeToString(), b"resulting_sum"
  )
  external_service_handle.release_unencrypted(
      client_count_val.SerializeToString(), b"resulting_client_count"
  )
  )",
                      /*kms_private_state=*/"", client_ids, client_data_dir);

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  tensorflow_federated::v0::Value released_sum;
  released_sum.ParseFromString(released_data["resulting_sum"]);
  ASSERT_THAT(released_sum.array().int32_list().value(),
              ::testing::ElementsAreArray({44, 52, 60}));
  tensorflow_federated::v0::Value released_client_count;
  released_client_count.ParseFromString(
      released_data["resulting_client_count"]);
  ASSERT_THAT(released_client_count.array().int32_list().value(),
              ::testing::ElementsAreArray({8}));

  auto released_state_changes =
      this->fake_data_read_write_service_.GetReleasedStateChanges();
  // There is no initial state.
  ASSERT_FALSE(
      released_state_changes["resulting_sum"].first.value().has_value());
  // The first release operation triggers a state change that should decrease
  // the number of remaining runs and increment the counter.
  BudgetState expected_first_release_budget;
  expected_first_release_budget.set_num_runs_remaining(kMaxNumRuns - 1);
  expected_first_release_budget.set_counter(1);
  ASSERT_EQ(released_state_changes["resulting_sum"].second.value(),
            expected_first_release_budget.SerializeAsString());

  ASSERT_TRUE(session_response.has_finalize());
}

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithModelLoading) {
  this->CreateSession(
      R"(
import os
import zipfile

import federated_language
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np

def trusted_program(input_provider, external_service_handle):
  zip_file_path = input_provider.get_filename_for_config_id('model1')
  model_path = os.path.join(os.path.dirname(zip_file_path), 'model1')
  with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(model_path)
  model = tff.learning.models.load_functional_model(model_path)

  def model_fn() -> tff.learning.models.VariableModel:
    return tff.learning.models.model_from_functional(model)

  learning_process = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=tff.learning.optimizers.build_sgdm(
          learning_rate=0.01
      ),
  )
  state = learning_process.initialize()

  state_val, _ = tff.framework.serialize_value(
      state,
      federated_language.framework.infer_type(
          state,
      ),
  )
  external_service_handle.release_unencrypted(
      state_val.SerializeToString(), b"result"
  )
  )",
      /*kms_private_state=*/"",
      /*client_ids=*/{}, /*client_data_dir=*/"",
      /*file_id_to_filepath=*/
      {{"model1", "testdata/model1.zip"}});

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(released_data["result"]);
  ASSERT_EQ(released_value.struct_().element().size(), 5);

  auto released_state_changes =
      this->fake_data_read_write_service_.GetReleasedStateChanges();
  // There is no initial state.
  ASSERT_FALSE(released_state_changes["result"].first.value().has_value());
  // The first release operation triggers a state change that should decrease
  // the number of remaining runs and increment the counter.
  BudgetState expected_first_release_budget;
  expected_first_release_budget.set_num_runs_remaining(kMaxNumRuns - 1);
  expected_first_release_budget.set_counter(1);
  ASSERT_EQ(released_state_changes["result"].second.value(),
            expected_first_release_budget.SerializeAsString());

  ASSERT_TRUE(session_response.has_finalize());
}

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithJax2tf) {
  this->CreateSession(R"(
import functools

import federated_language
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow_federated as tff

jax.config.update("jax_serialization_version", 8)

_jax2tf_convert_cpu_native = functools.partial(
    jax2tf.convert,
    native_serialization=True,
    native_serialization_platforms=['cpu'],
)

def trusted_program(input_provider, external_service_handle):

  data_type = federated_language.FederatedType(
      federated_language.TensorType(np.int32), federated_language.SERVER
  )

  @federated_language.federated_computation(data_type)
  def my_comp(x):

    @tff.tensorflow.computation
    def tf_comp(x):

      def jax_comp(x):
        return jnp.square(x)

      return _jax2tf_convert_cpu_native(jax_comp)(x)

    return federated_language.federated_map(tf_comp, x)

  result = my_comp(5)
  result_val, _ = tff.framework.serialize_value(
      result,
      federated_language.framework.infer_type(result),
  )
  external_service_handle.release_unencrypted(
      result_val.SerializeToString(), b"result"
  )
  )");

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  tensorflow_federated::v0::Value released_sum;
  released_sum.ParseFromString(released_data["result"]);
  ASSERT_THAT(released_sum.array().int32_list().value(),
              ::testing::ElementsAreArray({25}));

  auto released_state_changes =
      this->fake_data_read_write_service_.GetReleasedStateChanges();
  // There is no initial state.
  ASSERT_FALSE(released_state_changes["result"].first.value().has_value());
  // The first release operation triggers a state change that should decrease
  // the number of remaining runs and increment the counter.
  BudgetState expected_first_release_budget;
  expected_first_release_budget.set_num_runs_remaining(kMaxNumRuns - 1);
  expected_first_release_budget.set_counter(1);
  ASSERT_EQ(released_state_changes["result"].second.value(),
            expected_first_release_budget.SerializeAsString());

  ASSERT_TRUE(session_response.has_finalize());
}

}  // namespace

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee
