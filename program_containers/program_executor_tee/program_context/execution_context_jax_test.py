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
    self.computation_delegation_service = (
        fake_service_bindings_jax.FakeComputationDelegationService(
            self.worker_bns
        )
    )
    self.server = fake_service_bindings_jax.FakeServer(
        self.untrusted_root_port,
        None,
        self.computation_delegation_service,
    )
    self.server.start()

    self.context = execution_context.TrustedContext(
        compile_to_call_dominant,
        XLA_COMPUTATION_RUNNER_BINARY_PATH,
        self.outgoing_server_address,
        self.worker_bns,
        self.serialized_reference_values,
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

  async def test_execution_context_arg(self):
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

      # Change the cardinality of the inputs.
      result_1, result_2 = my_comp([1, 2, 3, 4], 10)
      self.assertEqual(result_1, 10)
      self.assertEqual(result_2, 10)

  async def test_execution_context_server_arg_only(self):
    with federated_language.framework.get_context_stack().install(self.context):
      server_state_type = federated_language.FederatedType(
          np.int32, federated_language.SERVER
      )

      @federated_language.federated_computation(server_state_type)
      def my_comp(server_state):
        return server_state

      result = my_comp(10)
    self.assertEqual(result, 10)

  async def test_execution_context_jax_computation(self):
    with federated_language.framework.get_context_stack().install(self.context):

      @jax_computation.jax_computation(np.int32)
      def my_comp(x):
        return x + 1

      result = my_comp(10)

    self.assertEqual(result, 11)


if __name__ == "__main__":
  absltest.main()
