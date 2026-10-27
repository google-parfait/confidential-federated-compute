# Copyright 2025 Google LLC.
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

import collections
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import compilers
import computation_runner_bindings_tensorflow
import fake_service_bindings_tensorflow
import federated_language
import numpy as np
import portpicker
from program_executor_tee.program_context import execution_context
import tensorflow as tf
import tensorflow_federated as tff


class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.worker_bns = []
    self.serialized_reference_values = b""
    self.data_read_write_service = (
        fake_service_bindings_tensorflow.FakeDataReadWriteService()
    )
    self.server = fake_service_bindings_tensorflow.FakeServer(
        self.untrusted_root_port, self.data_read_write_service, None
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  async def test_compiler_caching(self):
    mock_compiler = unittest.mock.Mock()
    context = execution_context.TrustedContext(
        mock_compiler,
        computation_runner_bindings_tensorflow.ComputationRunner,
        None,
        self.outgoing_server_address,
        self.worker_bns,
        self.serialized_reference_values,
    )

    client_data_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )

    @federated_language.federated_computation(client_data_type)
    def my_comp(client_data):
      return federated_language.federated_sum(client_data)

    # Test calling my_comp with two different args of different client
    # cardinality.
    my_comp_arg_1 = [1, 2]
    my_comp_arg_2 = [3, 4, 5]

    expected_result_1 = sum(my_comp_arg_1)
    expected_result_2 = sum(my_comp_arg_2)

    mock_compiler.return_value = compilers.compile_tf_to_call_dominant(my_comp)

    result_1 = context.invoke(my_comp, my_comp_arg_1)
    result_2 = context.invoke(my_comp, my_comp_arg_2)

    # The compilation helper function should only be called once due to
    # caching.
    mock_compiler.assert_called_once_with(my_comp)

    self.assertEqual(result_1, expected_result_1)
    self.assertEqual(result_2, expected_result_2)

  async def test_federated_sum(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )

    client_data_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(
        [client_data_type, server_state_type]
    )
    def my_comp(client_data, server_state):
      return federated_language.federated_sum(client_data), server_state

    result_1, result_2 = my_comp([1, 2], 10)
    self.assertEqual(result_1, 3)
    self.assertEqual(result_2, 10)

    # Change the cardinality of the inputs.
    result_1, result_2 = my_comp([1, 2, 3], 10)
    self.assertEqual(result_1, 6)
    self.assertEqual(result_2, 10)

  async def test_federated_eval_no_arg(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )

    @tff.tensorflow.computation
    def create_ten():
      return 10

    @federated_language.federated_computation
    def my_comp():
      return federated_language.federated_eval(
          create_ten, federated_language.SERVER
      )

    self.assertEqual(my_comp(), 10)

  async def test_server_arg_only(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(server_state_type)
    def my_comp(server_state):
      return server_state

    result = my_comp(10)
    self.assertEqual(result, 10)

  async def test_tf_computation(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )

    @tff.tensorflow.computation(np.int32)
    def my_comp(x):
      return x + 1

    result = my_comp(10)
    self.assertEqual(result, 11)

  async def test_tf_computation_returns_numpy_arrays_without_copies(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )

    @tff.tensorflow.computation
    def create_tensors():
      return (
          tf.reshape(tf.range(1000, dtype=tf.float32), [10, 100]),
          tf.constant([-1, 2, -3], tf.int8),
      )

    floats, ints = create_tensors()

    # The TensorFlow executor returns tensors in `content`, as a single chunk
    # that is the buffer of the tensor. So the NumPy arrays share that buffer,
    # and are read-only (as `tff.framework.deserialize_value` returns arrays for
    # `content`).
    for array in (floats, ints):
      self.assertIsInstance(array, np.ndarray)
      self.assertFalse(array.flags.writeable)
      self.assertEqual(type(array.base).__name__, "PyCapsule")
    self.assertEqual(floats.dtype, np.float32)
    np.testing.assert_array_equal(
        floats, np.arange(1000, dtype=np.float32).reshape([10, 100])
    )
    self.assertEqual(ints.dtype, np.int8)
    np.testing.assert_array_equal(ints, np.array([-1, 2, -3], np.int8))

  async def test_federated_sequence_round_trip(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
        )
    )
    sequence_type = federated_language.FederatedType(
        federated_language.SequenceType(
            federated_language.StructType([
                ("x", federated_language.TensorType(np.int32)),
                ("y", federated_language.TensorType(np.float32, [2])),
            ])
        ),
        federated_language.CLIENTS,
    )

    @federated_language.federated_computation(sequence_type)
    def identity(value):
      return value

    # Each client's sequence is a list of elements. None are empty, since the
    # TensorFlow executor cannot create a dataset from an empty sequence.
    value = [
        [{"x": 1, "y": [1.0, 2.0]}, {"x": 2, "y": [3.0, 4.0]}],
        [{"x": 3, "y": [5.0, 6.0]}],
    ]
    np.testing.assert_equal(identity(value), value)


class ExecutionContextDistributedTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.num_workers = 3
    self.worker_bns = [f"bns_address_{i}" for i in range(self.num_workers)]
    self.serialized_reference_values = b""
    self.data_read_write_service = (
        fake_service_bindings_tensorflow.FakeDataReadWriteService()
    )
    self.computation_delegation_service = (
        fake_service_bindings_tensorflow.FakeComputationDelegationService(
            self.worker_bns
        )
    )
    self.server = fake_service_bindings_tensorflow.FakeServer(
        self.untrusted_root_port,
        self.data_read_write_service,
        self.computation_delegation_service,
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  @parameterized.named_parameters(
      ("composing_executor", False),
      ("elastic_composing_executor", True),
  )
  async def test_federated_map_and_sum(self, use_elastic_composing_executor):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
            use_elastic_composing_executor=use_elastic_composing_executor,
        )
    )

    client_data_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @tff.tensorflow.computation(np.int32)
    def double(value):
      return value * 2

    @federated_language.federated_computation(
        [client_data_type, server_state_type]
    )
    def my_comp(client_data, server_state):
      client_data = federated_language.federated_map(double, client_data)
      return federated_language.federated_sum(client_data), server_state

    result_1, result_2 = my_comp([1, 2], 10)
    self.assertEqual(result_1, 6)
    self.assertEqual(result_2, 10)

    # Change the cardinality of the inputs.
    result_1, result_2 = my_comp([1, 2, 3, 4], 10)
    self.assertEqual(result_1, 20)
    self.assertEqual(result_2, 10)

  @parameterized.named_parameters(
      ("composing_executor", False),
      ("elastic_composing_executor", True),
  )
  async def test_federated_mean(self, use_elastic_composing_executor):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
            use_elastic_composing_executor=use_elastic_composing_executor,
        )
    )

    client_data_type = federated_language.FederatedType(
        np.float32, federated_language.CLIENTS
    )

    @federated_language.federated_computation(client_data_type)
    def my_comp(client_data):
      return federated_language.federated_mean(client_data)

    result = my_comp([1.0, 2.0, 3.0, 4.0])
    self.assertAlmostEqual(float(result), 2.5, places=5)

  @parameterized.named_parameters(
      ("composing_executor", False),
      ("elastic_composing_executor", True),
  )
  async def test_server_arg_only(self, use_elastic_composing_executor):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
            use_elastic_composing_executor=use_elastic_composing_executor,
        )
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(server_state_type)
    def my_comp(server_state):
      return server_state

    result = my_comp(10)
    self.assertEqual(result, 10)

  @parameterized.named_parameters(
      ("composing_executor", False),
      ("elastic_composing_executor", True),
  )
  async def test_tf_computation(self, use_elastic_composing_executor):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            computation_runner_bindings_tensorflow.ComputationRunner,
            None,
            self.outgoing_server_address,
            self.worker_bns,
            self.serialized_reference_values,
            use_elastic_composing_executor=use_elastic_composing_executor,
        )
    )

    @tff.tensorflow.computation(np.int32)
    def my_comp(x):
      return x + 1

    result = my_comp(10)
    self.assertEqual(result, 11)


class MergeableExecutionContextTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.num_workers = 3
    self.worker_bns = [f"bns_address_{i}" for i in range(self.num_workers)]
    self.serialized_reference_values = b""
    self.data_read_write_service = (
        fake_service_bindings_tensorflow.FakeDataReadWriteService()
    )
    self.computation_delegation_service = (
        fake_service_bindings_tensorflow.FakeComputationDelegationService(
            self.worker_bns
        )
    )
    self.server = fake_service_bindings_tensorflow.FakeServer(
        self.untrusted_root_port,
        self.data_read_write_service,
        self.computation_delegation_service,
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  def _create_context(self):
    return execution_context.TrustedContext(
        compilers.compile_tf_to_call_dominant,
        computation_runner_bindings_tensorflow.ComputationRunner,
        tff.backends.native.compile_to_mergeable_comp_form,
        self.outgoing_server_address,
        self.worker_bns,
        self.serialized_reference_values,
    )

  async def test_federated_sum(self):
    context = self._create_context()
    federated_language.framework.set_default_context(context)

    client_data_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(
        [client_data_type, server_state_type]
    )
    def my_comp(client_data, server_state):
      return federated_language.federated_sum(client_data), server_state

    result_1, result_2 = my_comp([1, 2], 10)
    self.assertEqual(result_1, 3)
    self.assertEqual(result_2, 10)

    # Change the cardinality of the inputs.
    result_1, result_2 = my_comp([1, 2, 3, 4], 10)
    self.assertEqual(result_1, 10)
    self.assertEqual(result_2, 10)

  async def test_learning_process_next(self):
    context = self._create_context()
    federated_language.framework.set_default_context(context)

    def decode_example(examples):
      parsed = tf.io.parse_example(
          examples,
          features={
              "x": tf.io.FixedLenFeature([2, 2], tf.float32),
              "y": tf.io.FixedLenFeature([2, 1], tf.float32),
          },
      )
      return [parsed["x"], parsed["y"]]

    @tff.tensorflow.computation(tf.TensorSpec(shape=[None], dtype=tf.string))
    def _client_input_preprocessing(
        serialized_inputs: tf.Tensor,
    ) -> tf.data.Dataset:
      dataset = tf.data.Dataset.from_tensor_slices(serialized_inputs)
      dataset = dataset.map(decode_example)
      return dataset

    inputs = tf.keras.Input(shape=(2,), dtype=tf.float32)
    outputs = tf.keras.layers.Dense(1, use_bias=True)(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss = tf.keras.losses.MeanSquaredError()
    input_spec = (
        tf.TensorSpec(shape=[2, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[2, 1], dtype=tf.float32),
    )

    def _metrics_constructor():
      return collections.OrderedDict({
          "loss": tf.keras.metrics.MeanSquaredError(),
          "num_examples": tff.learning.metrics.NumExamplesCounter(),
      })

    model = tff.learning.models.functional_model_from_keras(
        keras_model,
        loss_fn=loss,
        input_spec=input_spec,
        metrics_constructor=_metrics_constructor,
    )
    learning_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(
            learning_rate=0.01
        ),
    )
    learning_process = (
        tff.simulation.compose_dataset_computation_with_learning_process(
            _client_input_preprocessing, learning_process
        )
    )

    state = learning_process.initialize()

    # Create dummy client datasets with serialized tf.train.Example
    def make_serialized_example():
      feature = {
          "x": tf.train.Feature(
              float_list=tf.train.FloatList(value=[1.0, 2.0, 3.0, 4.0])
          ),
          "y": tf.train.Feature(
              float_list=tf.train.FloatList(value=[1.0, 2.0])
          ),
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      return example.SerializeToString()

    client_data = [
        tff.framework.serialize_value(
            np.array(
                [make_serialized_example() for _ in range(3)], dtype=np.object_
            ),
            federated_language.TensorType(np.str_, [3]),
        )[0].array
        for _ in range(3)
    ]
    new_state, metrics = learning_process.next(state, client_data)
    self.assertIsNotNone(new_state)
    self.assertIsNotNone(metrics)


if __name__ == "__main__":
  absltest.main()
