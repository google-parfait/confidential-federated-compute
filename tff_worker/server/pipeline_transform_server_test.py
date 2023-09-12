# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from concurrent import futures
from typing import List
import unittest
from fcp.protos.confidentialcompute import pipeline_transform_pb2
from fcp.protos.confidentialcompute import pipeline_transform_pb2_grpc
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2 as worker_pb2
from google.protobuf import any_pb2
import grpc
import portpicker
import tensorflow as tf
from tensorflow.core.framework import types_pb2
import tensorflow_federated as tff
from tff_worker.server import pipeline_transform_server
from tff_worker.server.testing import checkpoint_test_utils


# Define a simple computation that adds a broadcasted value to all floats in
# a single input tensor.
@tff.tf_computation(
    OrderedDict([('float', tff.TensorType(tf.float32, shape=[None]))]),
    tf.float32,
)
def tf_comp(
    example: OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> OrderedDict[str, tf.Tensor]:
  result = OrderedDict()
  for name, t in example.items():
    result[name] = tf.add(t, broadcasted_data)
  return result


# Define a federated computation using tf_comp to be performed per-client.
@tff.federated_computation(
    tff.type_at_clients(
        OrderedDict([('float', tff.TensorType(tf.float32, shape=[None]))])
    ),
    tff.type_at_clients(tf.float32, all_equal=True),
)
def client_work_comp(
    example: OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> OrderedDict[str, tf.Tensor]:
  return tff.federated_map(tf_comp, (example, broadcasted_data))


@tff.federated_computation(
    tff.type_at_server(tf.int32), tff.type_at_clients(tf.int32)
)
def aggregation_comp(state: int, value: List[int]) -> int:
  state_at_clients = tff.federated_broadcast(state)
  scaled_value = tff.federated_map(
      tff.tf_computation(lambda x, y: x * y), (value, state_at_clients)
  )
  summed_value = tff.federated_sum(scaled_value)
  return summed_value


class PipelineTransformServicerTest(unittest.TestCase):

  # Initialize a server on the local machine for testing purposes, because
  # grpcio_testing requires a newer grpcio version than what is required by
  # TFF.
  def setUp(self):
    self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pipeline_transform_pb2_grpc.add_PipelineTransformServicer_to_server(
        pipeline_transform_server.PipelineTransformServicer(), self._server
    )
    self._port = portpicker.pick_unused_port()
    self._server.add_insecure_port('[::]:%d' % self._port)
    self._server.start()
    self._channel = grpc.insecure_channel('localhost:%d' % self._port)
    self._stub = pipeline_transform_pb2_grpc.PipelineTransformStub(
        self._channel
    )

  def tearDown(self):
    self._server.stop(grace=None)
    self._server.wait_for_termination()

  def test_initialize_unimplemented(self):
    request = pipeline_transform_pb2.InitializeRequest()
    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Initialize(request)
    self.assertIs(e.exception.code(), grpc.StatusCode.UNIMPLEMENTED)

  def test_generate_nonces_unimplemented(self):
    request = pipeline_transform_pb2.GenerateNoncesRequest()
    with self.assertRaises(grpc.RpcError) as e:
      self._stub.GenerateNonces(request)
    self.assertIs(e.exception.code(), grpc.StatusCode.UNIMPLEMENTED)

  def test_configure_and_attest(self):
    config = worker_pb2.TffWorkerConfiguration()
    any_config = any_pb2.Any()
    any_config.Pack(config)
    request = pipeline_transform_pb2.ConfigureAndAttestRequest(
        configuration=any_config
    )
    self.assertEqual(
        pipeline_transform_pb2.ConfigureAndAttestResponse(),
        self._stub.ConfigureAndAttest(request),
    )

  def test_configure_and_attest_empty_config(self):
    request = pipeline_transform_pb2.ConfigureAndAttestRequest()

    with self.assertRaises(grpc.RpcError) as e:
      self._stub.ConfigureAndAttest(request)
    self.assertIs(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
    self.assertIn(
        'ConfigureAndAttestRequest must contain configuration.',
        str(e.exception),
    )

  def test_configure_and_attest_invalid_config(self):
    record = pipeline_transform_pb2.Record()
    any_config = any_pb2.Any()
    any_config.Pack(record)

    request = pipeline_transform_pb2.ConfigureAndAttestRequest(
        configuration=any_config
    )

    with self.assertRaises(grpc.RpcError) as e:
      self._stub.ConfigureAndAttest(request)
    self.assertIs(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
    self.assertIn(
        'ConfigureAndAttestRequest configuration must be a'
        ' tff_worker_configuration_pb2.TffWorkerConfiguration',
        str(e.exception),
    )

  def test_transform_without_configure(self):
    request = pipeline_transform_pb2.TransformRequest()
    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Transform(request)
    self.assertIs(e.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
    self.assertIn(
        'ConfigureAndAttest must be called before Transform', str(e.exception)
    )

  def test_transform_executes_client_work(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tff.type_at_clients(tf.float32, all_equal=True)
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.append(
        checkpoint_test_utils.column_config(
            name='float', data_type=types_pb2.DT_FLOAT
        )
    )
    config = worker_pb2.TffWorkerConfiguration(client_work=client_work_config)
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    transform_request.inputs.add().unencrypted_data = (
        checkpoint_test_utils.create_checkpoint_bytes(
            ['float'], [tf.constant([1.0, 2.0, 3.0])]
        )
    )

    serialized_expected_output, _ = tff.framework.serialize_value(
        [
            OrderedDict([
                ('float', tf.constant([11.0, 12.0, 13.0])),
            ])
        ],
        tff.types.at_clients(
            OrderedDict([
                ('float', tff.TensorType(tf.float32, shape=([None]))),
            ])
        ),
    )
    expected_response = pipeline_transform_pb2.TransformResponse()
    expected_response.outputs.add().unencrypted_data = (
        serialized_expected_output.SerializeToString()
    )

    self.assertEqual(expected_response, self._stub.Transform(transform_request))

  def test_transform_executes_aggregation(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.types.at_server(tf.int32)
    )
    config = worker_pb2.TffWorkerConfiguration()
    config.aggregation.serialized_client_to_server_aggregation_computation = (
        tff.framework.serialize_computation(
            aggregation_comp
        ).SerializeToString()
    )
    config.aggregation.serialized_temporary_state = (
        serialized_temp_state.SerializeToString()
    )
    config.aggregation.min_clients_in_aggregate = 3
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.types.at_clients(tf.int32)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.types.at_clients(tf.int32)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.types.at_clients(tf.int32)
    )
    transform_request.inputs.add().unencrypted_data = (
        serialized_input_a.SerializeToString()
    )
    transform_request.inputs.add().unencrypted_data = (
        serialized_input_b.SerializeToString()
    )
    transform_request.inputs.add().unencrypted_data = (
        serialized_input_c.SerializeToString()
    )

    serialized_expected_output, _ = tff.framework.serialize_value(
        tf.constant(2 * (5 + 6 + 7)), tff.types.at_server(tf.int32)
    )
    expected_response = pipeline_transform_pb2.TransformResponse()
    expected_response.outputs.add().unencrypted_data = (
        serialized_expected_output.SerializeToString()
    )
    self.assertEqual(expected_response, self._stub.Transform(transform_request))

  def test_transform_client_work_invalid_input(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tff.type_at_clients(tf.float32, all_equal=True)
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.append(
        checkpoint_test_utils.column_config(
            name='float', data_type=types_pb2.DT_FLOAT
        )
    )
    config = worker_pb2.TffWorkerConfiguration(client_work=client_work_config)
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    # The tensor name in the checkpoint doesn't match the name expected in the
    # configuration.
    transform_request.inputs.add().unencrypted_data = (
        checkpoint_test_utils.create_checkpoint_bytes(
            ['other_name'], [tf.constant([1.0, 2.0, 3.0])]
        )
    )

    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Transform(transform_request)
    self.assertIs(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
    self.assertIn(
        'TypeError when executing client work computation',
        str(e.exception),
    )

  def test_transform_aggregation_invalid_input(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.types.at_server(tf.int32)
    )
    config = worker_pb2.TffWorkerConfiguration()
    config.aggregation.serialized_client_to_server_aggregation_computation = (
        tff.framework.serialize_computation(
            aggregation_comp
        ).SerializeToString()
    )
    config.aggregation.serialized_temporary_state = (
        serialized_temp_state.SerializeToString()
    )
    config.aggregation.min_clients_in_aggregate = 3
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.types.at_clients(tf.int32)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.types.at_clients(tf.int32)
    )
    transform_request.inputs.add().unencrypted_data = (
        serialized_input_a.SerializeToString()
    )
    transform_request.inputs.add().unencrypted_data = (
        serialized_input_b.SerializeToString()
    )
    transform_request.inputs.add().unencrypted_data = b'invalid_input'
    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Transform(transform_request)
    self.assertIs(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
    self.assertIn(
        'TypeError when executing aggregation computation',
        str(e.exception),
    )

  def test_transform_client_work_wrong_num_inputs(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tff.type_at_clients(tf.float32, all_equal=True)
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.append(
        checkpoint_test_utils.column_config(
            name='float', data_type=types_pb2.DT_FLOAT
        )
    )
    config = worker_pb2.TffWorkerConfiguration(client_work=client_work_config)
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    # Add two inputs each containing a valid checkpoint. `client_work`
    # transforms expect only one input.
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        ['float'], [tf.constant([1.0, 2.0, 3.0])]
    )
    transform_request.inputs.add().unencrypted_data = unencrypted_data
    transform_request.inputs.add().unencrypted_data = unencrypted_data

    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Transform(transform_request)
    self.assertIs(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
    self.assertIn(
        'Exactly one input must be provided to a `client_work` transform but'
        ' got 2',
        str(e.exception),
    )


if __name__ == '__main__':
  unittest.main(verbosity=2)
