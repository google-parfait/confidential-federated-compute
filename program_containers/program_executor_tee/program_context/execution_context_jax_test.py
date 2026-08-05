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
    {"num_workers": 0, "use_elastic_composing_executor": False},
    {"num_workers": 2, "use_elastic_composing_executor": False},
    {"num_workers": 2, "use_elastic_composing_executor": True},
    {"num_workers": 3, "use_elastic_composing_executor": False},
    {"num_workers": 3, "use_elastic_composing_executor": True},
    {"num_workers": 4, "use_elastic_composing_executor": False},
    {"num_workers": 4, "use_elastic_composing_executor": True},
])
class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.context = None
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
        use_elastic_composing_executor=getattr(
            self, "use_elastic_composing_executor"
        ),
    )

  def tearDown(self):
    self.server.stop()
    if self.context:
      self.context.close()

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

      # 0 clients
      res_1, res_2 = my_comp([], 10)
      self.assertEqual(res_1, 0)
      self.assertEqual(res_2, 10)

      # 1 client
      res_1, res_2 = my_comp([42], 10)
      self.assertEqual(res_1, 42)
      self.assertEqual(res_2, 10)

      # 5 clients
      res_1, res_2 = my_comp([1, 2, 3, 4, 5], 10)
      self.assertEqual(res_1, 15)
      self.assertEqual(res_2, 10)

      # 100 clients: sum(1..100) = 5050
      res_1, res_2 = my_comp(list(range(1, 101)), 10)
      self.assertEqual(res_1, 5050)
      self.assertEqual(res_2, 10)

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

  async def test_execution_context_worker_failover(self):
    if self.num_workers < 2 or not getattr(
        self, "use_elastic_composing_executor", False
    ):
      return
    # Inject worker 1 failure mid-computation.
    self.computation_delegation_service.set_worker_failing("bns_address_1")
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

      # Even with worker 1 failing, the elastic executor reassigns all work to worker 0.
      result_1, result_2 = my_comp([1, 2, 3, 4], 10)
      self.assertEqual(result_1, 10)
      self.assertEqual(result_2, 10)

  async def test_execution_context_all_workers_failing(self):
    if self.num_workers < 2 or not getattr(
        self, "use_elastic_composing_executor", False
    ):
      return
    # Mark all workers failing.
    for bns in self.worker_bns:
      self.computation_delegation_service.set_worker_failing(bns)

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

      with self.assertRaises(Exception):
        my_comp([1, 2, 3, 4], 10)

  async def test_execution_context_federated_map(self):
    with federated_language.framework.get_context_stack().install(self.context):
      value_type = federated_language.TensorType(np.int32)
      client_data_type = federated_language.FederatedType(
          value_type, federated_language.CLIENTS
      )

      @jax_computation.jax_computation(np.int32)
      def add_one(x):
        return x + 1

      @federated_language.federated_computation(client_data_type)
      def map_comp(client_data):
        return federated_language.federated_map(add_one, client_data)

      # 1 client
      result_1 = map_comp([42])
      self.assertEqual(list(result_1), [43])

      # 5 clients
      result_5 = map_comp([1, 2, 3, 4, 5])
      self.assertEqual(list(result_5), [2, 3, 4, 5, 6])

      # 100 clients
      inputs_100 = list(range(1, 101))
      result_100 = map_comp(inputs_100)
      self.assertEqual(list(result_100), [x + 1 for x in inputs_100])

  async def test_execution_context_worker_failover_during_map(self):
    if self.num_workers < 2 or not getattr(
        self, "use_elastic_composing_executor", False
    ):
      return
    # Inject worker 1 failure.
    self.computation_delegation_service.set_worker_failing("bns_address_1")
    with federated_language.framework.get_context_stack().install(self.context):
      value_type = federated_language.TensorType(np.int32)
      client_data_type = federated_language.FederatedType(
          value_type, federated_language.CLIENTS
      )

      @jax_computation.jax_computation(np.int32)
      def add_one(x):
        return x + 1

      @federated_language.federated_computation(client_data_type)
      def map_comp(client_data):
        return federated_language.federated_map(add_one, client_data)

      # Even with worker 1 failing, the elastic executor reassigns work.
      result = map_comp([10, 20, 30, 40])
      self.assertEqual(list(result), [11, 21, 31, 41])

  async def test_execution_context_worker_recovery(self):
    if self.num_workers < 2 or not getattr(
        self, "use_elastic_composing_executor", False
    ):
      return
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

      # Round 1: all workers healthy.
      self.computation_delegation_service.reset_worker_call_counts()
      result_1, result_2 = my_comp([1, 2, 3, 4], 10)
      self.assertEqual(result_1, 10)
      self.assertEqual(result_2, 10)
      # Verify both workers completed requests successfully.
      self.assertGreater(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_0"
          ),
          0,
      )
      self.assertGreater(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_1"
          ),
          0,
      )

      # Round 2: worker 1 fails, surviving workers handle all work.
      self.computation_delegation_service.set_worker_failing("bns_address_1")
      self.computation_delegation_service.reset_worker_call_counts()
      result_1, result_2 = my_comp([1, 2, 3, 4], 10)
      self.assertEqual(result_1, 10)
      self.assertEqual(result_2, 10)
      # Worker 0 handled all the work. Worker 1 completed zero successful requests.
      self.assertGreater(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_0"
          ),
          0,
      )
      self.assertEqual(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_1"
          ),
          0,
      )

      # Round 3: worker 1 recovers. Confirm both workers participate.
      self.computation_delegation_service.clear_worker_failing("bns_address_1")
      self.computation_delegation_service.reset_worker_call_counts()
      result_1, result_2 = my_comp(list(range(1, 101)), 10)
      self.assertEqual(result_1, 5050)
      self.assertEqual(result_2, 10)
      # Verify both workers handled requests.
      self.assertGreater(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_0"
          ),
          0,
      )
      self.assertGreater(
          self.computation_delegation_service.get_worker_successful_call_count(
              "bns_address_1"
          ),
          0,
      )


if __name__ == "__main__":
  absltest.main()
