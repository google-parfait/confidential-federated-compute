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

from concurrent import futures
import unittest
from fcp.protos.confidentialcompute import pipeline_transform_pb2
from fcp.protos.confidentialcompute import pipeline_transform_pb2_grpc
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2
from google.protobuf import any_pb2
import grpc
import portpicker
import tensorflow as tf
import tensorflow_federated as tff
from tff_worker.server import pipeline_transform_server


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
    print('Starting Pipeline Transform server')
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
    config = tff_worker_configuration_pb2.TffWorkerConfiguration()
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

  def test_transform_client_work_unimplemented(self):
    config = tff_worker_configuration_pb2.TffWorkerConfiguration()
    config.client_work.MergeFrom(
        tff_worker_configuration_pb2.TffWorkerConfiguration.ClientWork()
    )
    any_config = any_pb2.Any()
    any_config.Pack(config)
    configure_and_attest_request = (
        pipeline_transform_pb2.ConfigureAndAttestRequest(
            configuration=any_config
        )
    )
    self._stub.ConfigureAndAttest(configure_and_attest_request)

    transform_request = pipeline_transform_pb2.TransformRequest()
    with self.assertRaises(grpc.RpcError) as e:
      self._stub.Transform(transform_request)
    self.assertIs(e.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
    self.assertIn(
        'Performing a `client_work` Transform is unimplemented',
        str(e.exception),
    )

  def test_transform_executes_aggregation(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    config = tff_worker_configuration_pb2.TffWorkerConfiguration()
    config.aggregation.serialized_client_to_server_aggregation_computation = (
        serialized_aggregation_comp.SerializeToString()
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
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
        tf.constant('The quick brown fox'), tf.string
    )
    expected_response = pipeline_transform_pb2.TransformResponse()
    expected_response.outputs.add().unencrypted_data = (
        serialized_expected_output.SerializeToString()
    )
    self.assertEqual(expected_response, self._stub.Transform(transform_request))

  def test_transform_invalid_input(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    config = tff_worker_configuration_pb2.TffWorkerConfiguration()
    config.aggregation.serialized_client_to_server_aggregation_computation = (
        serialized_aggregation_comp.SerializeToString()
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
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


if __name__ == '__main__':
  unittest.main(verbosity=2)
