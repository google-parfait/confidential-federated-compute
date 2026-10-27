# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for execution_context_helper."""

import asyncio
import collections
import unittest
from unittest import mock

from absl.testing import absltest
import federated_language
from federated_language_jax.computation import jax_computation
from google.protobuf import message
import jax.numpy as jnp
import numpy as np
from program_executor_tee.program_context import execution_context_helper
from program_executor_tee.program_context.cc import computation_runner_bindings_jax
import tensorflow_federated as tff


def _create_runner():
  return computation_runner_bindings_jax.ComputationRunner()


def _identity(type_spec):
  @federated_language.federated_computation(type_spec)
  def identity(value):
    return value

  return identity


class ExecutionContextHelperTest(absltest.TestCase):

  def test_execute_in_process_no_arg(self):
    @federated_language.federated_computation()
    def return_ten():
      return 10

    result = execution_context_helper.execute_in_process(
        _create_runner(), return_ten
    )
    self.assertEqual(result, 10)

  def test_execute_in_process_with_arg(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )

    result = execution_context_helper.execute_in_process(
        _create_runner(), _identity(client_type), [10, 20, 30]
    )
    self.assertEqual(result, [10, 20, 30])

  def test_execute_in_process_server_arg(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    result = execution_context_helper.execute_in_process(
        _create_runner(), _identity(server_type), 42
    )
    self.assertEqual(result, 42)

  def test_execute_in_process_mixed_arg(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation([client_type, server_type])
    def mixed_comp(client_val, server_val):
      return client_val, server_val

    result = execution_context_helper.execute_in_process(
        _create_runner(), mixed_comp, ([10, 20, 30, 40], 99)
    )
    self.assertEqual(result, ([10, 20, 30, 40], 99))

  def test_execute_in_process_selects_struct_element(self):
    @federated_language.federated_computation(
        federated_language.TensorType(np.int32),
        federated_language.TensorType(np.int32),
    )
    def second(x, y):
      del x  # Unused.
      return y

    result = execution_context_helper.execute_in_process(
        _create_runner(), second, (np.int32(10), np.int32(20))
    )
    self.assertEqual(result, 20)

  def test_execute_in_process_federated_all_equal_value(self):
    # The number of clients is inferred from the value that is not all-equal.
    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
        federated_language.FederatedType(
            np.int32, federated_language.CLIENTS, all_equal=True
        ),
    )
    def identity(x, y):
      return x, y

    result = execution_context_helper.execute_in_process(
        _create_runner(), identity, ([1, 2, 3], 5)
    )
    self.assertEqual(result, ([1, 2, 3], 5))

  def test_execute_in_process_with_federated_arg_and_result(self):
    TestValue = collections.namedtuple("TestValue", ["x", "y"])
    type_spec = federated_language.FederatedType(
        federated_language.StructWithPythonType(
            elements=[
                ("x", federated_language.TensorType(np.int32)),
                ("y", federated_language.TensorType(np.float32)),
            ],
            container_type=TestValue,
        ),
        federated_language.CLIENTS,
    )
    arg = [TestValue(x=1, y=2.0), TestValue(x=3, y=4.0)]
    result = execution_context_helper.execute_in_process(
        _create_runner(), _identity(type_spec), arg
    )
    self.assertEqual(result, arg)
    self.assertIsInstance(result[0], TestValue)

  def test_execute_in_process_string_client_value_larger_than_2gib(self):
    client_type = federated_language.FederatedType(
        federated_language.TensorType(np.str_), federated_language.CLIENTS
    )
    # Strings are still sent as serialized protos, and TFF fails to serialize a
    # string of 2 GiB or more, before the runner is called.
    with self.assertRaises(message.DecodeError):
      execution_context_helper.execute_in_process(
          _create_runner(), _identity(client_type), [b"\0" * 2**31]
      )

  def test_execute_in_process_client_values_larger_than_2gib(self):
    # Each client's tensor (800 MiB) is sent as a NumPy array, with no
    # serialization, and their total (2.4 GiB) is not limited.
    num_elements = 200 * 2**20
    client_type = federated_language.FederatedType(
        federated_language.TensorType(np.float32, [num_elements]),
        federated_language.CLIENTS,
    )
    arg = [
        np.arange(num_elements, dtype=np.float32) + client
        for client in range(3)
    ]

    result = execution_context_helper.execute_in_process(
        _create_runner(), _identity(client_type), arg
    )

    self.assertLen(result, 3)
    for client_result, client_arg in zip(result, arg):
      self.assertIsInstance(client_result, np.ndarray)
      self.assertEqual(client_result.dtype, np.float32)
      self.assertEqual(client_result.shape, (num_elements,))
      self.assertTrue(np.array_equal(client_result, client_arg))

  def test_execute_in_process_result_larger_than_2gib(self):
    num_elements = 2**29 + 1  # Just over 2 GiB of float32.

    @jax_computation.jax_computation
    def create_large_array():
      return jnp.arange(num_elements, dtype=np.float32)

    # The result is not serialized: the NumPy array owns the typed list of the
    # C++ result.
    result = execution_context_helper.execute_in_process(
        _create_runner(), create_large_array
    )

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.float32)
    self.assertEqual(result.shape, (num_elements,))
    self.assertTrue(result.flags.writeable)
    self.assertEqual(result[0], 0.0)
    self.assertEqual(result[12345], 12345.0)
    self.assertEqual(result[2**28], float(2**28))
    self.assertEqual(result[-1], float(2**29))

  def test_execute_rejects_fn_that_is_not_a_serialized_value(self):
    comp_pb, _ = tff.framework.serialize_value(
        _identity(federated_language.TensorType(np.int32))
    )
    runner = _create_runner()
    # `fn` must be bytes, so even an unserialized proto is rejected.
    for value in (None, "not bytes", comp_pb):
      with self.assertRaisesRegex(TypeError, "is not an instance of 'bytes'"):
        runner.execute(value, None)
    with self.assertRaisesRegex(
        ValueError, "Failed to parse serialized tensorflow_federated.v0.Value"
    ):
      runner.execute(b"not a proto", None)

  def test_contains_clients_placement_federated_clients(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    self.assertTrue(
        execution_context_helper.contains_clients_placement(client_type)
    )

  def test_contains_clients_placement_federated_server(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    self.assertFalse(
        execution_context_helper.contains_clients_placement(server_type)
    )

  def test_contains_clients_placement_struct_with_clients(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    struct_type = federated_language.StructType([client_type, server_type])
    self.assertTrue(
        execution_context_helper.contains_clients_placement(struct_type)
    )

  def test_contains_clients_placement_struct_without_clients(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    struct_type = federated_language.StructType(
        [server_type, federated_language.TensorType(np.float32)]
    )
    self.assertFalse(
        execution_context_helper.contains_clients_placement(struct_type)
    )

  def test_contains_clients_placement_function_type_in_parameter(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(client_type, server_type)
    self.assertTrue(
        execution_context_helper.contains_clients_placement(func_type)
    )

  def test_contains_clients_placement_function_type_in_result(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    func_type = federated_language.FunctionType(server_type, client_type)
    self.assertTrue(
        execution_context_helper.contains_clients_placement(func_type)
    )

  def test_contains_clients_placement_function_type_no_parameter(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    func_type = federated_language.FunctionType(None, client_type)
    self.assertTrue(
        execution_context_helper.contains_clients_placement(func_type)
    )

  def test_contains_clients_placement_function_type_no_clients(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(server_type, server_type)
    self.assertFalse(
        execution_context_helper.contains_clients_placement(func_type)
    )

  def test_contains_clients_placement_sequence_type(self):
    seq_type = federated_language.SequenceType(
        federated_language.TensorType(np.int32)
    )
    self.assertFalse(
        execution_context_helper.contains_clients_placement(seq_type)
    )

  def test_contains_clients_placement_tensor_type(self):
    tensor_type = federated_language.TensorType(np.int32)
    self.assertFalse(
        execution_context_helper.contains_clients_placement(tensor_type)
    )


class ResilientSubroundsTest(unittest.IsolatedAsyncioTestCase):

  async def test_run_resilient_subrounds_success(self):
    async def _coro(arg, ctx):
      await asyncio.sleep(0.001)
      return (arg, f"res_{arg}_on_{ctx}")

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    worker_contexts = ["ctx1", "ctx2"]
    args = [10, 20, 30]
    results, _ = await execution_context_helper.run_resilient_subrounds(
        mock_task, args, worker_contexts
    )
    results_dict = dict(results)
    self.assertEqual(len(results), 3)
    self.assertIn(results_dict[10], ("res_10_on_ctx1", "res_10_on_ctx2"))
    self.assertIn(results_dict[20], ("res_20_on_ctx1", "res_20_on_ctx2"))
    self.assertIn(results_dict[30], ("res_30_on_ctx1", "res_30_on_ctx2"))

  async def test_run_resilient_subrounds_retryable_error_requeues(self):
    attempts = {}

    async def _coro(arg, ctx):
      attempts[arg] = attempts.get(arg, 0) + 1
      if attempts[arg] == 1 and arg == 10:
        raise ConnectionError("Temporary disconnect")
      return (arg, arg * 2)

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    worker_contexts = ["ctx1", "ctx2"]
    args = [10, 20]
    results, _ = await execution_context_helper.run_resilient_subrounds(
        mock_task, args, worker_contexts
    )
    results_dict = dict(results)
    self.assertEqual(results_dict[10], 20)
    self.assertEqual(results_dict[20], 40)
    self.assertEqual(attempts[10], 2)

  async def test_run_resilient_subrounds_max_retries_exceeded(self):
    async def _coro(arg, ctx):
      raise ConnectionError("Persistent disconnect")

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    worker_contexts = ["ctx1", "ctx2", "ctx3"]
    args = [10]
    with self.assertRaisesRegex(RuntimeError, "failed after 2 retries"):
      await execution_context_helper.run_resilient_subrounds(
          mock_task,
          args,
          worker_contexts,
          max_retries_per_subround=2,
      )

  async def test_run_resilient_subrounds_all_workers_fail(self):
    dead_contexts = set()

    async def _coro(arg, ctx):
      dead_contexts.add(ctx)
      raise ConnectionError("Worker crashed")

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    worker_contexts = ["ctx1", "ctx2"]
    args = [10, 20, 30]
    with self.assertRaisesRegex(RuntimeError, "All execution contexts failed"):
      await execution_context_helper.run_resilient_subrounds(
          mock_task,
          args,
          worker_contexts,
          max_retries_per_subround=5,
      )
    self.assertEqual(dead_contexts, {"ctx1", "ctx2"})

  async def test_run_resilient_subrounds_with_postprocessing_hook(self):
    async def _coro(arg, ctx):
      return arg * 2

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    async def hook(acc, val, ctx):
      if acc is None:
        return val
      return acc + val

    worker_contexts = ["ctx1", "ctx2"]
    args = [1, 2, 3, 4, 5]
    result, _ = await execution_context_helper.run_resilient_subrounds(
        mock_task,
        args,
        worker_contexts,
        initial_result=None,
        postprocessing=hook,
    )
    self.assertEqual(result, 30)

  async def test_run_resilient_subrounds_decouples_dispatch_from_postprocessing(
      self,
  ):
    """Verifies subround 2 is dispatched before subround 1 finishes postprocessing."""
    timeline = []

    async def _coro(arg, ctx):
      timeline.append(f"worker_dispatched_{arg}")
      return arg

    def mock_task(arg, ctx):
      return asyncio.create_task(_coro(arg, ctx))

    async def slow_postprocessing(acc, val, ctx):
      timeline.append(f"merge_started_{val}")
      await asyncio.sleep(0.05)
      timeline.append(f"merge_finished_{val}")
      return (acc or 0) + val

    # With only 1 worker context, subround 2 could not be dispatched under the
    # old synchronous loop until subround 1's postprocessing finished.
    result, _ = await execution_context_helper.run_resilient_subrounds(
        mock_task,
        [1, 2],
        ["ctx1"],
        initial_result=None,
        postprocessing=slow_postprocessing,
    )

    self.assertEqual(result, 3)
    # Confirm worker 2 was dispatched while subround 1 was still in postprocessing.
    self.assertLess(
        timeline.index("worker_dispatched_2"),
        timeline.index("merge_finished_1"),
    )



class RunnerAsyncContextTest(unittest.IsolatedAsyncioTestCase):
  """Tests for RunnerAsyncContext.invoke."""

  def _make_fake_comp(self, type_signature):
    comp = mock.MagicMock()
    comp.type_signature = type_signature
    return comp

  async def test_invoke_routes_to_worker_bns_for_clients_placement(self):
    """Computations with @CLIENTS should execute on the context's worker."""
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(client_type, server_type)
    comp = self._make_fake_comp(func_type)
    runner = mock.MagicMock()

    with mock.patch.object(
        execution_context_helper, "execute_in_process", return_value=99
    ) as mock_execute:
      ctx = execution_context_helper.RunnerAsyncContext(
          runner, worker_bns="worker_0"
      )
      result = await ctx.invoke(comp, arg=[1, 2, 3])

    self.assertEqual(result, 99)
    mock_execute.assert_called_once_with(runner, comp, [1, 2, 3], "worker_0")

  async def test_invoke_routes_empty_bns_for_server_only(self):
    """Computations without @CLIENTS should execute locally."""
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(server_type, server_type)
    comp = self._make_fake_comp(func_type)
    runner = mock.MagicMock()

    with mock.patch.object(
        execution_context_helper, "execute_in_process", return_value=99
    ) as mock_execute:
      ctx = execution_context_helper.RunnerAsyncContext(
          runner, worker_bns="worker_0"
      )
      result = await ctx.invoke(comp)

    self.assertEqual(result, 99)
    mock_execute.assert_called_once_with(runner, comp, None, "")

  async def test_invoke_wraps_error_in_runtime_error(self):
    """Errors should be wrapped in RuntimeError with the target worker."""
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(client_type, server_type)
    comp = self._make_fake_comp(func_type)

    with mock.patch.object(
        execution_context_helper,
        "execute_in_process",
        side_effect=ValueError("deadline exceeded"),
    ):
      ctx = execution_context_helper.RunnerAsyncContext(
          mock.MagicMock(), worker_bns="worker_0"
      )
      with self.assertRaisesRegex(
          RuntimeError,
          "Request to computation runner failed with error: deadline exceeded"
          r" \(target worker: 'worker_0'\)",
      ):
        await ctx.invoke(comp, arg=[1, 2, 3])


if __name__ == "__main__":
  absltest.main()
