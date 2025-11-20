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
import compilers
import fake_service_bindings_tensorflow
import federated_language
import numpy as np
import portpicker
from program_executor_tee.program_context import execution_context
import tensorflow_federated as tff


TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH = (
    "computation_runner_binary_tensorflow"
)


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
        TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_tf_execution_context(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_tf_execution_context_no_arg(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_execution_context_server_arg_only(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_execution_context_tf_computation(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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


class ExecutionContextDistributedTest(unittest.IsolatedAsyncioTestCase):

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

  async def test_execution_context(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_execution_context_server_arg_only(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            compilers.compile_tf_to_call_dominant,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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

  async def test_execution_context_tf_computation(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedContext(
            lambda x: x,
            TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH,
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


if __name__ == "__main__":
  absltest.main()
