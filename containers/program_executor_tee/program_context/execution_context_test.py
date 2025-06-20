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

import functools
import threading
import unittest
from absl.testing import absltest
from containers.program_executor_tee.program_context import compilers
from containers.program_executor_tee.program_context import execution_context
from containers.program_executor_tee.program_context.cc import computation_runner_bindings
from containers.program_executor_tee.program_context.cc import fake_computation_delegation_service_bindings as fake_service_bindings
from fcp.confidentialcompute.python import compiler
from fcp.protos.confidentialcompute import computation_delegation_pb2_grpc
import federated_language
from federated_language_jax.computation import jax_computation
import grpc
import numpy as np
import portpicker
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

  def setUp(self):
    self.port = portpicker.pick_unused_port()
    self.assertIsNotNone(self.port, "Failed to pick an unused port.")

    # Create 4 workers. The first worker is the server, and the other 3 are
    # child workers.
    self.worker_bns = [
        "bns_address_1",
        "bns_address_2",
        "bns_address_3",
        "bns_address_4",
    ]

    self.service_impl = fake_service_bindings.FakeComputationDelegationService(
        self.worker_bns
    )
    self.server = fake_service_bindings.FakeServer(self.port, self.worker_bns)

    self.server_runner_thread = threading.Thread(
        target=self.server.start,
        args=(),
        daemon=True,
    )
    self.server_runner_thread.start()
    self.channel = grpc.insecure_channel(self.server.get_address())
    self.stub = computation_delegation_pb2_grpc.ComputationDelegationStub(
        self.channel
    )

  def tearDown(self):
    self.server.stop()
    self.server_runner_thread.join(timeout=5)

  def computation_delegation_stub_proxy(self, request):
    result = computation_runner_bindings.ComputationDelegationResult()
    try:
      response = self.stub.Execute(request)
      result.response = response
    except grpc.RpcError as e:
      result.status = e
    return result

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

    @jax_computation.jax_computation
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
    runner = computation_runner_bindings.ComputationRunner(
        self.worker_bns, self.computation_delegation_stub_proxy, "fake_attester"
    )

    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            functools.partial(
                compiler.to_composed_tee_form,
                num_client_workers=len(self.worker_bns) - 1,
            ),
            runner.invoke_comp,
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
    result_1, result_2 = await my_comp([1, 2, 3, 4], 10)
    self.assertEqual(result_1, 10)
    self.assertEqual(result_2, 10)


if __name__ == "__main__":
  absltest.main()
