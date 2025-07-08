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
import unittest
from absl.testing import absltest
from containers.program_executor_tee.program_context import compilers
from containers.program_executor_tee.program_context import execution_context
from containers.program_executor_tee.program_context import test_helpers
from containers.program_executor_tee.program_context.cc import fake_service_bindings as fake_service_bindings
from fcp.confidentialcompute.python import compiler
import federated_language
from federated_language_jax.computation import jax_computation
import numpy as np
import portpicker
import tensorflow_federated as tff


class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.worker_bns = []
    self.attester_id = ""
    self.data_read_write_service = (
        fake_service_bindings.FakeDataReadWriteService()
    )
    self.server = fake_service_bindings.FakeServer(
        self.untrusted_root_port, self.data_read_write_service, None
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  async def test_compiler_caching(self):
    mock_compiler = unittest.mock.Mock()
    context = execution_context.TrustedAsyncContext(
        mock_compiler,
        self.outgoing_server_address,
        self.worker_bns,
        self.attester_id,
        test_helpers.parse_read_response_fn,
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

    result_1 = await context.invoke(my_comp, my_comp_arg_1)
    result_2 = await context.invoke(my_comp, my_comp_arg_2)

    # The compilation helper function should only be called once due to
    # caching.
    mock_compiler.assert_called_once_with(my_comp)

    self.assertEqual(result_1, expected_result_1)
    self.assertEqual(result_2, expected_result_2)

  async def test_tf_execution_context(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            compilers.compile_tf_to_call_dominant,
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
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

  async def test_tf_execution_context_data_pointer_arg(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            compilers.compile_tf_to_call_dominant,
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
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

    self.data_read_write_service.store_plaintext_message(
        "client_1",
        test_helpers.create_array_value(
            1, client_data_type.member
        ).SerializeToString(),
    )
    self.data_read_write_service.store_plaintext_message(
        "client_2",
        test_helpers.create_array_value(
            2, client_data_type.member
        ).SerializeToString(),
    )
    client_1_data = test_helpers.create_data_value(
        "client_1", "mykey", client_data_type.member
    ).computation
    client_2_data = test_helpers.create_data_value(
        "client_2", "mykey", client_data_type.member
    ).computation

    result_1, result_2 = await my_comp([client_1_data, client_2_data], 10)
    self.assertEqual(result_1, 3)
    self.assertEqual(result_2, 10)

  async def test_tf_execution_context_no_arg(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            lambda x: x,
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
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

    self.assertEqual(await my_comp(), 10)

  async def test_tf_execution_context_jax_computation(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            lambda x: x,
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
        )
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
        "Request to computation runner failed with error:"
        " `TensorFlowExecutor::CreateValueComputation`",
        str(context.exception),
    )


class ExecutionContextDistributedTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"

    # Create 4 workers. The first worker is the server, and the other 3 are child workers.
    self.worker_bns = [
        "bns_address_1",
        "bns_address_2",
        "bns_address_3",
        "bns_address_4",
    ]
    self.attester_id = "fake_attester"
    self.data_read_write_service = (
        fake_service_bindings.FakeDataReadWriteService()
    )
    self.computation_delegation_service = (
        fake_service_bindings.FakeComputationDelegationService(self.worker_bns)
    )
    self.server = fake_service_bindings.FakeServer(
        self.untrusted_root_port,
        self.data_read_write_service,
        self.computation_delegation_service,
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  async def test_execution_context(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            functools.partial(
                compiler.to_composed_tee_form,
                num_client_workers=len(self.worker_bns) - 1,
            ),
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
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

  async def test_execution_context_data_pointer_arg(self):
    federated_language.framework.set_default_context(
        execution_context.TrustedAsyncContext(
            functools.partial(
                compiler.to_composed_tee_form,
                num_client_workers=len(self.worker_bns) - 1,
            ),
            self.outgoing_server_address,
            self.worker_bns,
            self.attester_id,
            test_helpers.parse_read_response_fn,
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

    self.data_read_write_service.store_plaintext_message(
        "client_1",
        test_helpers.create_array_value(
            1, client_data_type.member
        ).SerializeToString(),
    )
    self.data_read_write_service.store_plaintext_message(
        "client_2",
        test_helpers.create_array_value(
            2, client_data_type.member
        ).SerializeToString(),
    )
    client_1_data = test_helpers.create_data_value(
        "client_1", "mykey", client_data_type.member
    ).computation
    client_2_data = test_helpers.create_data_value(
        "client_2", "mykey", client_data_type.member
    ).computation

    result_1, result_2 = await my_comp([client_1_data, client_2_data], 10)
    self.assertEqual(result_1, 3)
    self.assertEqual(result_2, 10)


if __name__ == "__main__":
  absltest.main()
