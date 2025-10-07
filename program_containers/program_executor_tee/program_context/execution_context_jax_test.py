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
import federated_language
from federated_language_jax.computation import jax_computation
import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized_class
import portpicker
from program_executor_tee.program_context import execution_context
from program_executor_tee.program_context import test_helpers
from program_executor_tee.program_context.cc import fake_service_bindings_jax
import tensorflow_federated as tff


XLA_COMPUTATION_RUNNER_BINARY_PATH = (
    "program_executor_tee/program_context/cc/computation_runner_binary_xla"
)


def compile_to_call_dominant(
    comp: federated_language.framework.ConcreteComputation,
) -> federated_language.framework.ConcreteComputation:
  """Compile a computation to run on the program executor TEE."""
  comp_bb = tff.framework.to_call_dominant(comp.to_building_block())
  return federated_language.framework.ConcreteComputation.from_building_block(
      comp_bb
  )


def build_federated_sum_comp() -> federated_language.Computation:
  value_type = federated_language.TensorType(np.int32)
  client_data_type = federated_language.FederatedType(
      value_type, federated_language.CLIENTS
  )

  @jax_computation.jax_computation
  def create_zero():
    return jnp.zeros(shape=[], dtype=np.int32)

  @jax_computation.jax_computation(value_type, value_type)
  def add(a, b):
    return jax.tree_util.tree_map(jnp.add, a, b)

  @jax_computation.jax_computation
  def identity(x):
    return x

  @federated_language.federated_computation(client_data_type)
  def federated_sum(client_values):
    return federated_language.federated_aggregate(
        value=client_values,
        zero=create_zero(),
        accumulate=add,
        merge=add,
        report=identity,
    )

  return federated_sum


@parameterized_class([
    {"num_workers": 0},
    {"num_workers": 2},
    {"num_workers": 3},
    {"num_workers": 4},
])
class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.worker_bns = [f"bns_address_{i}" for i in range(self.num_workers)]
    self.serialized_reference_values = b""
    self.data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    self.computation_delegation_service = (
        fake_service_bindings_jax.FakeComputationDelegationService(
            self.worker_bns
        )
    )
    self.server = fake_service_bindings_jax.FakeServer(
        self.untrusted_root_port,
        self.data_read_write_service,
        self.computation_delegation_service,
    )
    self.server.start()

    self.context = execution_context.TrustedContext(
        compile_to_call_dominant,
        XLA_COMPUTATION_RUNNER_BINARY_PATH,
        self.outgoing_server_address,
        self.worker_bns,
        self.serialized_reference_values,
        test_helpers.parse_read_response_fn,
    )

  def tearDown(self):
    self.server.stop()

  async def test_execution_context_no_arg(self):
    with federated_language.framework.get_context_stack().install(self.context):

      @jax_computation.jax_computation
      def comp():
        return 10

      result = comp()

    self.assertEqual(result, 10)

  async def test_execution_contest_arg(self):
    with federated_language.framework.get_context_stack().install(self.context):
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
        return build_federated_sum_comp()(client_data), server_state

      result_1, result_2 = my_comp([1, 2, 3], 10)
      self.assertEqual(result_1, 6)
      self.assertEqual(result_2, 10)

  async def test_execution_contest_data_pointer_arg(self):
    with federated_language.framework.get_context_stack().install(self.context):
      client_data_type = federated_language.FederatedType(
          np.int32, federated_language.CLIENTS
      )
      server_state_type = federated_language.FederatedType(
          np.int32, federated_language.SERVER
      )

      @jax_computation.jax_computation
      def abs_diff(x, y):
        return jnp.abs(x - y)

      @federated_language.federated_computation(
          [client_data_type, server_state_type]
      )
      def my_comp(client_data, server_state):
        server_state_at_client = federated_language.federated_broadcast(
            server_state
        )
        shifted_client_values = federated_language.federated_map(
            abs_diff,
            federated_language.federated_zip(
                [server_state_at_client, client_data]
            ),
        )
        return build_federated_sum_comp()(shifted_client_values)

      self.data_read_write_service.store_plaintext_message(
          "client_1",
          test_helpers.create_array_value(
              1, client_data_type.member
          ).SerializeToString(),
      )
      self.data_read_write_service.store_plaintext_message(
          "client_2",
          test_helpers.create_array_value(
              8, client_data_type.member
          ).SerializeToString(),
      )
      client_1_data = test_helpers.create_data_value(
          "client_1", "mykey", client_data_type.member
      ).computation
      client_2_data = test_helpers.create_data_value(
          "client_2", "mykey", client_data_type.member
      ).computation

      result = my_comp([client_1_data, client_2_data, client_1_data], 5)

    # abs(1-5) + abs(8-5) + abs(1-5) = 4+3+4 = 11
    self.assertEqual(result, 11)

    # Check that each uri was requested only once.
    self.assertEqual(
        self.data_read_write_service.get_read_request_uris(),
        ["client_1", "client_2"],
    )


if __name__ == "__main__":
  absltest.main()
