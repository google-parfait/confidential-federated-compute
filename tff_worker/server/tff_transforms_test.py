# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
from typing import List, Tuple
import unittest
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2 as worker_pb2
import tensorflow as tf
from tensorflow.core.framework import types_pb2
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2
import tff_transforms
from tff_worker.server.testing import checkpoint_test_utils

_NAMES = ['float_a', 'int_a', 'float_b', 'string_b']
_FLOAT_A = tf.constant([1.0, 2.0])
_INT_A = tf.constant([1, 2])
_FLOAT_B = tf.constant([3.0, 4.0])
_STRING_B = tf.constant(['The', 'quick'])
_TENSORS = [_FLOAT_A, _INT_A, _FLOAT_B, _STRING_B]


@tff.tf_computation(
    OrderedDict([
        ('float_a', tff.TensorType(tf.float32, shape=[None])),
        ('int_a', tff.TensorType(tf.int32, shape=[None])),
        ('float_b', tff.TensorType(tf.float32, shape=[None])),
        ('string_b', tff.TensorType(tf.string, shape=[None])),
    ]),
    tf.float32,
)
def tf_comp(
    example: OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> OrderedDict[str, tf.Tensor]:
  result = OrderedDict()
  for name, t in example.items():
    if 'float' in name:
      result[name] = tf.add(t, broadcasted_data)
  return result


@tff.federated_computation(
    tff.FederatedType(
        OrderedDict([
            ('float_a', tff.TensorType(tf.float32, shape=[None])),
            ('int_a', tff.TensorType(tf.int32, shape=[None])),
            ('float_b', tff.TensorType(tf.float32, shape=[None])),
            ('string_b', tff.TensorType(tf.string, shape=[None])),
        ]),
        tff.CLIENTS,
    ),
    tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
)
def client_work_comp(
    example: OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> OrderedDict[str, tf.Tensor]:
  return tff.federated_map(tf_comp, (example, broadcasted_data))


@tff.federated_computation(
    tff.FederatedType(tf.int32, tff.SERVER),
    tff.FederatedType(tf.int32, tff.CLIENTS),
)
def aggregation_comp(
    state: int, value: List[int]
) -> tff.templates.MeasuredProcessOutput:
  state_at_clients = tff.federated_broadcast(state)
  scaled_value = tff.federated_map(
      tff.tf_computation(lambda x, y: x * y), (value, state_at_clients)
  )
  summed_value = tff.federated_sum(scaled_value)
  return tff.templates.MeasuredProcessOutput(
      state=state, result=summed_value, measurements=summed_value
  )


def column_config(
    name: str, data_type: types_pb2.DataType
) -> (
    worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint.FedSqlColumn
):
  return worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint.FedSqlColumn(
      name=name, data_type=data_type
  )


def get_checkpoint_config_matching_computation() -> (
    worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint
):
  checkpoint_config = (
      worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint()
  )
  checkpoint_config.fed_sql_columns.extend([
      column_config(name='float_a', data_type=types_pb2.DT_FLOAT),
      column_config(name='int_a', data_type=types_pb2.DT_INT32),
      column_config(name='float_b', data_type=types_pb2.DT_FLOAT),
      column_config(name='string_b', data_type=types_pb2.DT_STRING),
  ])
  return checkpoint_config


class TffTransformsTest(unittest.TestCase):

  def test_client_work(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    serialized_client_work = tff.framework.serialize_computation(
        client_work_comp
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=serialized_client_work.SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    result_bytes = tff_transforms.perform_client_work(
        client_work_config, unencrypted_data
    )
    result_proto = executor_pb2.Value.FromString(result_bytes)
    (serialized_expected_result, _) = tff.framework.serialize_value(
        [
            OrderedDict([
                ('float_a', tf.constant([11.0, 12.0])),
                ('float_b', tf.constant([13.0, 14.0])),
            ])
        ],
        tff.FederatedType(
            OrderedDict([
                ('float_a', tff.TensorType(tf.float32, shape=([None]))),
                ('float_b', tff.TensorType(tf.float32, shape=([None]))),
            ]),
            tff.CLIENTS,
        ),
    )
    self.assertEqual(serialized_expected_result, result_proto)
    self.assertEqual(
        serialized_client_work.type.function.result.federated,
        result_proto.federated.type,
    )

  def test_client_work_unparseable_computation(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=b'unparseable',
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )

    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn('Could not parse serialized computation', str(e.exception))

  def test_client_work_unparseable_broadcasted_data(self):
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=b'unparseable',
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Could not parse serialized broadcasted data', str(e.exception)
    )

  def test_client_work_client_input_tensor_name_mismatch(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # Add a tensor to the checkpoint that has an unexpected name.
    names = ['float_a', 'int_a', 'other_name', 'string_b']
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Could not parse FedSql checkpoint with expected columns',
        str(e.exception),
    )

  def test_client_work_client_input_tensor_type_mismatch(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # The third tensor will have tf.string datatype but the FedSql config
    # expects it to be a float.
    tensors = [_FLOAT_A, _INT_A, tf.constant(['thequick']), _STRING_B]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, tensors
    )

    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Could not parse FedSql checkpoint with expected columns',
        str(e.exception),
    )

  def test_client_work_client_input_tensor_count_mismatch(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # One of the expected tensors in the FedSql configuration is not present in
    # the input checkpoint.
    names = ['float_a', 'int_a', 'float_b']
    tensors = [_FLOAT_A, _INT_A, _FLOAT_B]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Could not parse FedSql checkpoint with expected columns',
        str(e.exception),
    )

  def test_client_work_client_input_tensor_shape_mismatch(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    names = ['float_a', 'int_a', 'float_b', 'string_b']
    # The third tensor has more elements than the rest, but all the tensor
    # shapes in the checkpoint should match.
    float_b = tf.constant([3.0, 4.0, 5.0, 6.0])
    tensors = [_FLOAT_A, _INT_A, float_b, _STRING_B]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Could not parse FedSql checkpoint with expected columns',
        str(e.exception),
    )

  def test_client_work_scalars_in_checkpoint(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # Providing scalars in the checkpoint instead of 1-dimensional tensors
    # should succeed.
    tensors = [
        tf.constant(1.0),
        tf.constant(1),
        tf.constant(3.0),
        tf.constant('The'),
    ]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn('All tensors must be one-dimensional', str(e.exception))

  def test_client_work_client_input_tensor_dims_mismatch(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    names = ['float_a', 'int_a', 'float_b', 'string_b']
    # The third tensor has more dimensions than the rest, but all the tensor
    # shapes in the checkpoint should match.
    float_b = tf.constant([[3.0, 4.0], [5.0, 6.0]])
    tensors = [_FLOAT_A, _INT_A, float_b, _STRING_B]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn('All tensors must be one-dimensional', str(e.exception))

  def test_client_work_client_input_tensor_name_mismatch_with_computation(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # The third tensor will have a name which matches the expected FedSql
    # checkpoint configuration, but doesn't match the expected name in the
    # computation.
    names = ['float_a', 'int_a', 'some_other_name', 'string_b']
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.extend([
        column_config(name='float_a', data_type=types_pb2.DT_FLOAT),
        column_config(name='int_a', data_type=types_pb2.DT_INT32),
        # The third tensor will have a name which doesn't match the expected
        # name in the computation.
        column_config(name='some_other_name', data_type=types_pb2.DT_FLOAT),
        column_config(name='string_b', data_type=types_pb2.DT_STRING),
    ])
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'FedSql checkpoint specification incompatible with client work'
        ' computation client input type',
        str(e.exception),
    )

  def test_client_work_client_input_tensor_type_mismatch_with_computation(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    names = ['float_a', 'int_a', 'float_b', 'string_b']
    float_a = tf.constant([1.0, 2.0])
    int_a = tf.constant([1, 2])
    # The third tensor will have type INT32, which matches the expected
    # FedSql checkpoint configuration, but doesn't match the expected
    # computation input type.
    float_b = tf.constant([3, 4])
    string_b = tf.constant(['The', 'quick'])
    tensors = [float_a, int_a, float_b, string_b]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.extend([
        column_config(name='float_a', data_type=types_pb2.DT_FLOAT),
        column_config(name='int_a', data_type=types_pb2.DT_INT32),
        # The third tensor will have type INT32, which matches the checkpoint,
        # but doesn't match the expected computation input type.
        column_config(name='float_b', data_type=types_pb2.DT_INT32),
        column_config(name='string_b', data_type=types_pb2.DT_STRING),
    ])
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'FedSql checkpoint specification incompatible with client work'
        ' computation client input type',
        str(e.exception),
    )

  def test_client_work_client_input_tensor_count_mismatch_with_computation(
      self,
  ):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    # One of the expected tensors in the computation input spec is not present
    # in the input checkpoint or the FedSql configuration.
    names = ['float_a', 'int_a', 'float_b']
    float_a = tf.constant([1.0, 2.0])
    int_a = tf.constant([1, 2])
    float_b = tf.constant([3.0, 4.0])
    tensors = [float_a, int_a, float_b]
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        names, tensors
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.fed_sql_columns.extend([
        column_config(name='float_a', data_type=types_pb2.DT_FLOAT),
        column_config(name='int_a', data_type=types_pb2.DT_INT32),
        column_config(name='float_b', data_type=types_pb2.DT_FLOAT),
    ])
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Number of elements computation expects in client input (4) does'
        ' not match the number of FedSql columns provided by configuration (3)',
        str(e.exception),
    )

  def test_client_work_computation_wrong_num_args(self):
    @tff.tf_computation(
        OrderedDict([
            ('float_a', tff.TensorType(tf.float32, shape=[None])),
            ('int_a', tff.TensorType(tf.int32, shape=[None])),
            ('float_b', tff.TensorType(tf.float32, shape=[None])),
            ('string_b', tff.TensorType(tf.string, shape=[None])),
        ])
    )
    def client_work_comp_wrong_num_args(
        example: OrderedDict[str, tf.Tensor]
    ) -> OrderedDict[str, tf.Tensor]:
      result = OrderedDict()
      for name, t in example.items():
        if 'float' in name:
          result[name] = tf.add(t, 1.0)
      return result

    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp_wrong_num_args
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn('Unexpected number of elements', str(e.exception))

  def test_client_work_computation_client_input_type_not_federated(self):
    # This test will use a tensorflow computation that expects a regular float
    # input, not a float with an @CLIENTS placement.
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tf.float32
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        # Use the raw tensorflow computation rather than the computation
        # wrapped in a federated map.
        serialized_client_work_computation=tff.framework.serialize_computation(
            tf_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Expected client work computation client input type to be an '
        'instance of type',
        str(e.exception),
    )

  def test_client_work_computation_client_input_type_member_not_struct(self):
    @tff.tf_computation(
        tff.TensorType(tf.float32, shape=[None]),
        tf.float32,
    )
    def comp_not_struct(
        example: tf.Tensor, broadcasted_data: float
    ) -> tf.Tensor:
      return tf.add(example, broadcasted_data)

    @tff.federated_computation(
        tff.FederatedType(tff.TensorType(tf.float32, shape=[None]), tff.CLIENTS),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    def client_work_comp_not_struct(
        example: tf.Tensor, broadcasted_data: float
    ) -> tf.Tensor:
      return tff.federated_map(comp_not_struct, (example, broadcasted_data))

    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0),
        tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True),
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp_not_struct
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Expected client work computation client input type member to be an '
        'instance of type',
        str(e.exception),
    )

  def test_client_work_computation_wrong_broadcast_type(self):
    # The computation expects that the broadcasted data is a float, but provide
    # a string.
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant('a string'),
        tff.FederatedType(tf.string, tff.CLIENTS, all_equal=True),
    )
    unencrypted_data = checkpoint_test_utils.create_checkpoint_bytes(
        _NAMES, _TENSORS
    )
    client_work_config = worker_pb2.TffWorkerConfiguration.ClientWork(
        serialized_client_work_computation=tff.framework.serialize_computation(
            client_work_comp
        ).SerializeToString(),
        serialized_broadcasted_data=(
            serialized_broadcasted_data.SerializeToString()
        ),
    )
    client_work_config.fed_sql_tensorflow_checkpoint.CopyFrom(
        get_checkpoint_config_matching_computation()
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.perform_client_work(client_work_config, unencrypted_data),
    self.assertIn(
        'Type `string@CLIENTS` is not assignable to type `float32@CLIENTS`',
        str(e.exception),
    )

  def test_aggregate(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.FederatedType(tf.int32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    result_bytes = tff_transforms.aggregate(
        aggregation_config,
        [
            serialized_input_a.SerializeToString(),
            serialized_input_b.SerializeToString(),
            serialized_input_c.SerializeToString(),
        ],
    )
    result_proto = executor_pb2.Value.FromString(result_bytes)
    (serialized_expected_result, _) = tff.framework.serialize_value(
        tff.templates.MeasuredProcessOutput(
            state=tf.constant(2),
            result=tf.constant(2 * (5 + 6 + 7)),
            measurements=(tf.constant(2 * (5 + 6 + 7))),
        ),
        tff.StructType([
            (
                'state',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()), tff.SERVER
                ),
            ),
            (
                'result',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()), tff.SERVER
                ),
            ),
            (
                'measurements',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()), tff.SERVER
                ),
            ),
        ]),
    )
    self.assertEqual(serialized_expected_result, result_proto)

  def test_aggregate_structs(self):
    @tff.federated_computation(
        tff.FederatedType(tf.string, tff.SERVER),
        OrderedDict([
            (
                'weights',
                tff.FederatedType(
                    tff.TensorType(tf.float32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
            (
                'counts',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
        ]),
    )
    def aggregation_comp(
        state: str, value: OrderedDict[str, List[tf.Tensor]]
    ) -> tf.float32:
      return (
          tff.federated_mean(value.weights),
          tff.federated_sum(value.counts),
          state,
      )

    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('hello world'), tff.FederatedType(tf.string, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        OrderedDict(
            [('weights', [tf.constant(3.0)]), ('counts', [tf.constant(5)])]
        ),
        OrderedDict([
            (
                'weights',
                tff.FederatedType(
                    tff.TensorType(tf.float32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
            (
                'counts',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
        ]),
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        OrderedDict(
            [('weights', [tf.constant(7.0)]), ('counts', [tf.constant(6)])]
        ),
        OrderedDict([
            (
                'weights',
                tff.FederatedType(
                    tff.TensorType(tf.float32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
            (
                'counts',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
        ]),
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        OrderedDict(
            [('weights', [tf.constant(2.0)]), ('counts', [tf.constant(8)])]
        ),
        OrderedDict([
            (
                'weights',
                tff.FederatedType(
                    tff.TensorType(tf.float32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
            (
                'counts',
                tff.FederatedType(
                    tff.TensorType(tf.int32, shape=()),
                    tff.CLIENTS,
                    all_equal=False,
                ),
            ),
        ]),
    )
    result_bytes = tff_transforms.aggregate(
        aggregation_config,
        [
            serialized_input_a.SerializeToString(),
            serialized_input_b.SerializeToString(),
            serialized_input_c.SerializeToString(),
        ],
    )
    result_proto = executor_pb2.Value.FromString(result_bytes)
    (serialized_expected_result, _) = tff.framework.serialize_value(
        ((3.0 + 7.0 + 2.0) / 3.0, (5 + 6 + 8), 'hello world'),
        tff.StructType([
            tff.FederatedType(tff.TensorType(tf.float32, shape=()), tff.SERVER),
            tff.FederatedType(tff.TensorType(tf.int32, shape=()), tff.SERVER),
            tff.FederatedType(tff.TensorType(tf.string, shape=()), tff.SERVER),
        ]),
    )
    self.assertEqual(serialized_expected_result, result_proto)

  def test_aggregate_unnamed_structs(self):
    @tff.federated_computation(
        tff.FederatedType(tf.string, tff.SERVER),
        tff.types.StructType([
            tff.FederatedType(
                tff.TensorType(tf.float32, shape=()),
                tff.CLIENTS,
                all_equal=False,
            ),
            tff.FederatedType(
                tff.TensorType(tf.int32, shape=()), tff.CLIENTS, all_equal=False
            ),
        ]),
    )
    def aggregation_comp(
        state: str, value: Tuple[List[tf.Tensor]]
    ) -> tf.float32:
      return (tff.federated_mean(value[0]), tff.federated_sum(value[1]), state)

    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('hello world'), tff.FederatedType(tf.string, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        ([tf.constant(3.0)], [tf.constant(5)]),
        tff.types.StructType([
            tff.FederatedType(
                tff.TensorType(tf.float32, shape=()),
                tff.CLIENTS,
                all_equal=False,
            ),
            tff.FederatedType(
                tff.TensorType(tf.int32, shape=()), tff.CLIENTS, all_equal=False
            ),
        ]),
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        ([tf.constant(7.0)], [tf.constant(6)]),
        tff.types.StructType([
            tff.FederatedType(
                tff.TensorType(tf.float32, shape=()),
                tff.CLIENTS,
                all_equal=False,
            ),
            tff.FederatedType(
                tff.TensorType(tf.int32, shape=()), tff.CLIENTS, all_equal=False
            ),
        ]),
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        ([tf.constant(2.0)], [tf.constant(8)]),
        tff.types.StructType([
            tff.FederatedType(
                tff.TensorType(tf.float32, shape=()),
                tff.CLIENTS,
                all_equal=False,
            ),
            tff.FederatedType(
                tff.TensorType(tf.int32, shape=()), tff.CLIENTS, all_equal=False
            ),
        ]),
    )
    result_bytes = tff_transforms.aggregate(
        aggregation_config,
        [
            serialized_input_a.SerializeToString(),
            serialized_input_b.SerializeToString(),
            serialized_input_c.SerializeToString(),
        ],
    )
    result_proto = executor_pb2.Value.FromString(result_bytes)
    (serialized_expected_result, _) = tff.framework.serialize_value(
        ((3.0 + 7.0 + 2.0) / 3.0, (5 + 6 + 8), 'hello world'),
        tff.types.StructType([
            tff.FederatedType(tff.TensorType(tf.float32, shape=()), tff.SERVER),
            tff.FederatedType(tff.TensorType(tf.int32, shape=()), tff.SERVER),
            tff.FederatedType(tff.TensorType(tf.string, shape=()), tff.SERVER),
        ]),
    )
    self.assertEqual(serialized_expected_result, result_proto)

  def test_aggregate_unparseable_computation(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.FederatedType(tf.int32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=b'invalid',
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn('Could not parse serialized computation', str(e.exception))

  def test_aggregate_unparseable_state(self):
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=b'invalid',
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Could not parse serialized temporary state', str(e.exception)
    )

  def test_aggregate_unparseable_client_input(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.FederatedType(tf.int32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              b'invalid_client_input',
          ],
      )
    self.assertIn('Could not parse serialized client input', str(e.exception))

  def test_aggregate_computation_wrong_num_args(self):
    # Create a computation that doesn't take in any server state.
    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def aggregation_comp(value: List[int]):
      scaled_value = tff.federated_map(
          tff.tf_computation(lambda x: x * 2), value
      )
      summed_value = tff.federated_sum(scaled_value)
      return tff.templates.MeasuredProcessOutput(
          state=summed_value, result=summed_value, measurements=summed_value
      )

    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=b'invalid',
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Expected aggregation computation input type', str(e.exception)
    )

  def test_aggregate_computation_wrong_temp_state_type(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(1.0), tff.FederatedType(tf.float32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant(7)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Type `float32@SERVER` is not assignable to type `int32@SERVER`',
        str(e.exception),
    )

  def test_aggregate_computation_wrong_client_input_type(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.FederatedType(tf.int32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        [tf.constant(5)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        [tf.constant(6)], tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        [tf.constant('input')], tff.FederatedType(tf.string, tff.CLIENTS)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Type `{string}@CLIENTS` is not assignable to type `{int32}@CLIENTS`',
        str(e.exception),
    )

  def test_aggregate_computation_client_input_type_all_equal_fails(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(2), tff.FederatedType(tf.int32, tff.SERVER)
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    # Client inputs where all_equal is True are invalid. TFF expects the client
    # input to have all_equal=False.
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant(5), tff.FederatedType(tf.int32, tff.CLIENTS, all_equal=True)
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant(6), tff.FederatedType(tf.int32, tff.CLIENTS, all_equal=True)
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant(7), tff.FederatedType(tf.int32, tff.CLIENTS, all_equal=True)
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          aggregation_config,
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Expected client input to be an instance of type',
        str(e.exception),
    )


if __name__ == '__main__':
  unittest.main(verbosity=2)
