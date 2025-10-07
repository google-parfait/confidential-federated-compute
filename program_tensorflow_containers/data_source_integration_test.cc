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
    BuildClientCheckpointFromInts;
using ::confidential_federated_compute::program_executor_tee::
    ProgramExecutorTeeSessionTest;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;

TYPED_TEST_SUITE(
    ProgramExecutorTeeSessionTest,
    ::testing::Types<TensorflowProgramExecutorTeeConfidentialTransform>);

TYPED_TEST(ProgramExecutorTeeSessionTest, ProgramWithDataSource) {
  std::vector<std::string> client_ids = {"client1", "client2", "client3",
                                         "client4"};
  std::string client_data_dir = "data_dir";
  std::string tensor_name = "output_tensor_name";
  for (int i = 0; i < client_ids.size(); i++) {
    CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
        client_data_dir + "/" + client_ids[i],
        BuildClientCheckpointFromInts({1 + i * 3, 2 + i * 3, 3 + i * 3},
                                      tensor_name)));
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

async def trusted_program(input_provider, release_manager):

  data_source = min_sep_data_source.MinSepDataSource(
      min_sep=2,
      client_ids=input_provider.client_ids,
      client_data_directory=input_provider.client_data_directory,
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

  await release_manager.release(server_state['sum'], "resulting_sum")
  await release_manager.release(server_state['client_count'], "resulting_client_count")
  )",
                      client_ids, client_data_dir);

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto write_call_args = this->fake_data_read_write_service_.GetWriteCallArgs();
  ASSERT_EQ(write_call_args.size(), 2);
  ASSERT_EQ(write_call_args[0].size(), 1);
  auto sum_write_request = write_call_args[0][0];
  ASSERT_EQ(sum_write_request.first_request_metadata().unencrypted().blob_id(),
            "resulting_sum");
  ASSERT_TRUE(sum_write_request.commit());
  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(sum_write_request.data());
  ASSERT_THAT(released_value.array().int32_list().value(),
              ::testing::ElementsAreArray({44, 52, 60}));

  ASSERT_EQ(write_call_args[1].size(), 1);
  auto count_write_request = write_call_args[1][0];
  ASSERT_EQ(
      count_write_request.first_request_metadata().unencrypted().blob_id(),
      "resulting_client_count");
  ASSERT_TRUE(count_write_request.commit());
  released_value.ParseFromString(count_write_request.data());
  ASSERT_THAT(released_value.array().int32_list().value(),
              ::testing::ElementsAreArray({8}));

  ASSERT_TRUE(session_response.has_finalize());
}

}  // namespace

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee
