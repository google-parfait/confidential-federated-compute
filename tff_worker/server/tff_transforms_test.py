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
from typing import List
import unittest
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2 as worker_pb2
import tensorflow as tf
from tensorflow.core.framework import types_pb2
import tensorflow_federated as tff
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
def client_work_comp(
    example: OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> OrderedDict[str, tf.Tensor]:
  result = OrderedDict()
  for name, t in example.items():
    if 'float' in name:
      result[name] = tf.add(t, broadcasted_data)
  return result


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
        tf.constant(10.0), tf.float32
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
    serialized_expected_output, _ = tff.framework.serialize_value(
        OrderedDict([
            ('float_a', tf.constant([11.0, 12.0])),
            ('float_b', tf.constant([13.0, 14.0])),
        ]),
        OrderedDict([
            ('float_a', tff.TensorType(tf.float32, shape=(2))),
            ('float_b', tff.TensorType(tf.float32, shape=(2))),
        ]),
    )
    self.assertEqual(
        serialized_expected_output.SerializeToString(),
        tff_transforms.perform_client_work(
            client_work_config, unencrypted_data
        ),
    )

  def test_client_work_unparseable_computation(self):
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        tf.constant(10.0), tf.float32
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
        'Number of elements expected in client input by computation (4) does'
        ' not match number of FedSql columns provided by configuration (3)',
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
        tf.constant(10.0), tf.float32
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

  def test_client_work_computation_client_input_type_not_struct(self):
    @tff.tf_computation(
        tff.TensorType(tf.float32, shape=[None]),
        tf.float32,
    )
    def client_work_comp_not_struct(
        example: tf.Tensor, broadcasted_data: float
    ) -> tf.Tensor:
      return tf.add(example, broadcasted_data)

    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant(10.0), tf.float32
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
        'Expected client work computation client input type to be an instance'
        ' of type',
        str(e.exception),
    )

  def test_client_work_computation_wrong_broadcast_type(self):
    # The computation expects that the broadcasted data is a float, but provide
    # a string.
    serialized_broadcasted_data, _ = tff.framework.serialize_value(
        tf.constant('a string'), tf.string
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
        'Type `string` is not assignable to type `float32`', str(e.exception)
    )

  def test_aggregate(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    serialized_expected_output, _ = tff.framework.serialize_value(
        tf.constant('The quick brown fox'), tf.string
    )
    self.assertEqual(
        serialized_expected_output.SerializeToString(),
        tff_transforms.aggregate(
            aggregation_config,
            [
                serialized_input_a.SerializeToString(),
                serialized_input_b.SerializeToString(),
                serialized_input_c.SerializeToString(),
            ],
        ),
    )

  def test_aggregate_unparseable_computation(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=b'invalid',
        serialized_temporary_state=serialized_temp_state.SerializeToString(),
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
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
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    aggregation_config = worker_pb2.TffWorkerConfiguration.Aggregation(
        serialized_client_to_server_aggregation_computation=(
            tff.framework.serialize_computation(
                aggregation_comp
            ).SerializeToString()
        ),
        serialized_temporary_state=b'invalid_temp_state',
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
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
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
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
    aggregation_comp = tff.tf_computation(
        lambda x: tf.strings.join([x, 'fox'], separator=' ', name=None),
        (tf.string),
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
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
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(1.0), tf.float32
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
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
        'Type `float32` is not assignable to type `string`', str(e.exception)
    )

  def test_aggregate_computation_wrong_client_input_type(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
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
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant(1.0), tf.float32
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
        'Type `float32` is not assignable to type `string`', str(e.exception)
    )


if __name__ == '__main__':
  unittest.main(verbosity=2)
