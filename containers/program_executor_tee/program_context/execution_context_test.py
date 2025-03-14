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

import unittest
from absl.testing import absltest
from containers.program_executor_tee.program_context import compilers
from containers.program_executor_tee.program_context import execution_context
from containers.program_executor_tee.program_context.cc import computation_runner_bindings
import federated_language
import numpy as np
import tensorflow_federated as tff


class ExecutionContextText(unittest.IsolatedAsyncioTestCase):

  async def test_invoke(self):
    mock_compiler = unittest.mock.Mock()
    mock_invoke = unittest.mock.Mock()
    context = execution_context.TrustedAsyncContext(mock_compiler, mock_invoke)

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

    expected_serialized_arg_1, _ = tff.framework.serialize_value(
        my_comp_arg_1, client_data_type
    )
    expected_serialized_arg_2, _ = tff.framework.serialize_value(
        my_comp_arg_2, client_data_type
    )
    expected_result_1 = sum(my_comp_arg_1)
    expected_result_2 = sum(my_comp_arg_2)
    expected_serialized_result_1, _ = tff.framework.serialize_value(
        expected_result_1, client_data_type.member
    )
    expected_serialized_result_2, _ = tff.framework.serialize_value(
        expected_result_2, client_data_type.member
    )

    mock_compiler.return_value = my_comp
    mock_invoke.side_effect = [
        expected_serialized_result_1,
        expected_serialized_result_2,
    ]

    result_1 = await context.invoke(my_comp, my_comp_arg_1)
    result_2 = await context.invoke(my_comp, my_comp_arg_2)

    # The compilation helper function should only be called once due to
    # caching.
    mock_compiler.assert_called_once_with(my_comp)

    serialized_comp, _ = tff.framework.serialize_value(my_comp)
    mock_invoke.assert_has_calls([
        unittest.mock.call(
            len(my_comp_arg_1), serialized_comp, expected_serialized_arg_1
        ),
        unittest.mock.call(
            len(my_comp_arg_2), serialized_comp, expected_serialized_arg_2
        ),
    ])

    self.assertEqual(result_1, expected_result_1)
    self.assertEqual(result_2, expected_result_2)

  async def test_invoke_no_arg(self):
    mock_compiler = unittest.mock.Mock()
    mock_invoke = unittest.mock.Mock()
    context = execution_context.TrustedAsyncContext(mock_compiler, mock_invoke)

    @federated_language.federated_computation
    def my_comp():
      return 10

    expected_result = 10
    expected_serialized_result, _ = tff.framework.serialize_value(
        expected_result, federated_language.TensorType(np.int32)
    )

    mock_compiler.return_value = my_comp
    mock_invoke.return_value = expected_serialized_result

    result = await context.invoke(my_comp, None)

    mock_compiler.assert_called_once_with(my_comp)

    serialized_comp, _ = tff.framework.serialize_value(my_comp)
    mock_invoke.assert_called_once_with(0, serialized_comp, None)

    self.assertEqual(result, expected_result)


class ExecutionContextIntegrationTest(unittest.IsolatedAsyncioTestCase):

  async def test_tf_execution_context_no_workers(self):
    runner = computation_runner_bindings.ComputationRunner([])
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            compilers.compile_tf_to_call_dominant, runner.invoke_comp
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

    result_1, result_2 = await my_comp([1, 2], 10)
    self.assertEqual(result_1, 3)
    self.assertEqual(result_2, 10)

    # Change the cardinality of the inputs.
    result_1, result_2 = await my_comp([1, 2, 3], 10)
    self.assertEqual(result_1, 6)
    self.assertEqual(result_2, 10)

  async def test_tf_execution_context_no_workers_no_arg(self):
    runner = computation_runner_bindings.ComputationRunner([])
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(lambda x: x, runner.invoke_comp)
    )

    @tff.tensorflow.computation
    def create_ten():
      return 10

    @federated_language.federated_computation
    def my_comp():
      return federated_language.federated_eval(
          create_ten, federated_language.SERVER
      )

    self.assertEqual(await my_comp(), 10)

  async def test_tf_execution_context_no_workers_jax_computation(self):
    runner = computation_runner_bindings.ComputationRunner([])
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(lambda x: x, runner.invoke_comp)
    )

    @tff.jax.computation
    def create_ten_jax():
      return 10

    @federated_language.federated_computation
    def my_comp():
      return federated_language.federated_eval(
          create_ten_jax, federated_language.SERVER
      )

    with self.assertRaises(RuntimeError) as context:
      await my_comp()
    self.assertIn(
        "TensorFlowExecutor::CreateValueComputation",
        str(context.exception),
    )

  async def test_execution_context_with_workers(self):
    runner = computation_runner_bindings.ComputationRunner(["bns_address"])
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(lambda x: x, runner.invoke_comp)
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

    with self.assertRaises(RuntimeError) as context:
      await my_comp([1, 2], 10)
    self.assertEqual(
        "Failed to execute computation: Distributed execution is not supported"
        " yet.",
        str(context.exception),
    )


if __name__ == "__main__":
  absltest.main()
