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
import unittest
from unittest import mock

from absl.testing import absltest
from fcp.protos.confidentialcompute import computation_delegation_pb2
from fcp.protos.confidentialcompute import tff_config_pb2
import federated_language
from google.protobuf import any_pb2
import numpy as np
from program_executor_tee.program_context import execution_context_helper
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


class ExecutionContextHelperTest(absltest.TestCase):

  def test_create_computation_request_no_arg(self):
    @federated_language.federated_computation()
    def return_ten():
      return 10

    req = execution_context_helper.create_computation_request(return_ten)
    self.assertTrue(
        req.computation.Is(tff_config_pb2.TffSessionConfig.DESCRIPTOR)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertIsInstance(session_config.function, executor_pb2.Value)
    self.assertEqual(session_config.num_clients, 0)
    self.assertFalse(session_config.HasField("initial_arg"))

  def test_create_computation_request_with_arg(self):
    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def sum_comp(val):
      return federated_language.federated_sum(val)

    req = execution_context_helper.create_computation_request(
        sum_comp, arg=[10, 20, 30]
    )
    self.assertTrue(
        req.computation.Is(tff_config_pb2.TffSessionConfig.DESCRIPTOR)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertIsInstance(session_config.function, executor_pb2.Value)
    self.assertEqual(session_config.num_clients, 3)
    self.assertTrue(session_config.HasField("initial_arg"))
    self.assertIsInstance(session_config.initial_arg, executor_pb2.Value)

  def test_create_computation_request_server_arg(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(server_type)
    def identity_comp(val):
      return val

    req = execution_context_helper.create_computation_request(
        identity_comp, arg=42
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertEqual(session_config.num_clients, 0)

  def test_create_computation_request_mixed_arg(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation([client_type, server_type])
    def mixed_comp(client_val, server_val):
      return federated_language.federated_sum(client_val), server_val

    req = execution_context_helper.create_computation_request(
        mixed_comp, arg=([10, 20, 30, 40], 99)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertEqual(session_config.num_clients, 4)

  def test_unpack_and_deserialize_computation_response(self):
    val_proto, _ = tff.framework.serialize_value(
        np.int32(42), federated_language.TensorType(np.int32)
    )
    any_proto = any_pb2.Any()
    any_proto.Pack(val_proto)
    resp = computation_delegation_pb2.ComputationResponse(result=any_proto)
    res = execution_context_helper.unpack_and_deserialize_computation_response(
        resp, federated_language.TensorType(np.int32)
    )
    self.assertEqual(res, 42)

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


class RunnerAsyncContextTest(unittest.IsolatedAsyncioTestCase):
  """Tests for RunnerAsyncContext.invoke."""

  def _make_stub(self, response=None, side_effect=None):
    stub = mock.MagicMock()
    if side_effect is not None:
      stub.Execute.side_effect = side_effect
    else:
      stub.Execute.return_value = response
    return stub

  def _make_fake_comp(self, type_signature):
    comp = mock.MagicMock()
    comp.type_signature = type_signature
    return comp

  async def test_invoke_routes_to_worker_bns_for_clients_placement(self):
    """Computations with @CLIENTS should set worker_bns on the request."""
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(client_type, server_type)
    comp = self._make_fake_comp(func_type)

    val_proto, _ = tff.framework.serialize_value(
        np.int32(99), federated_language.TensorType(np.int32)
    )
    any_proto = any_pb2.Any()
    any_proto.Pack(val_proto)
    response = computation_delegation_pb2.ComputationResponse(result=any_proto)
    stub = self._make_stub(response=response)

    with mock.patch.object(
        execution_context_helper,
        "create_computation_request",
    ) as mock_create:
      mock_req = computation_delegation_pb2.ComputationRequest()
      mock_create.return_value = mock_req

      ctx = execution_context_helper.RunnerAsyncContext(
          stub, worker_bns="worker_0"
      )
      await ctx.invoke(comp, arg=[1, 2, 3])

      self.assertEqual(mock_req.worker_bns, "worker_0")
      stub.Execute.assert_called_once_with(mock_req)

  async def test_invoke_routes_empty_bns_for_server_only(self):
    """Computations without @CLIENTS should set worker_bns to empty."""
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(server_type, server_type)
    comp = self._make_fake_comp(func_type)

    val_proto, _ = tff.framework.serialize_value(
        np.int32(99), federated_language.TensorType(np.int32)
    )
    any_proto = any_pb2.Any()
    any_proto.Pack(val_proto)
    response = computation_delegation_pb2.ComputationResponse(result=any_proto)
    stub = self._make_stub(response=response)

    with mock.patch.object(
        execution_context_helper,
        "create_computation_request",
    ) as mock_create:
      mock_req = computation_delegation_pb2.ComputationRequest()
      mock_create.return_value = mock_req

      ctx = execution_context_helper.RunnerAsyncContext(
          stub, worker_bns="worker_0"
      )
      await ctx.invoke(comp)

      self.assertEqual(mock_req.worker_bns, "")
      stub.Execute.assert_called_once_with(mock_req)

  async def test_invoke_wraps_grpc_error_in_runtime_error(self):
    """gRPC errors should be wrapped in RuntimeError with details."""
    import grpc

    class FakeRpcError(grpc.RpcError):

      def details(self):
        return "deadline exceeded"

    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(None, server_type)
    comp = self._make_fake_comp(func_type)

    stub = self._make_stub(side_effect=FakeRpcError())

    with mock.patch.object(
        execution_context_helper,
        "create_computation_request",
        return_value=computation_delegation_pb2.ComputationRequest(),
    ):
      ctx = execution_context_helper.RunnerAsyncContext(stub)
      with self.assertRaisesRegex(
          RuntimeError, "Request to computation runner failed"
      ):
        await ctx.invoke(comp)

  async def test_invoke_wraps_deserialization_error_in_runtime_error(self):
    """Non-gRPC exceptions should be wrapped in RuntimeError."""
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    func_type = federated_language.FunctionType(None, server_type)
    comp = self._make_fake_comp(func_type)

    # Return a response with an invalid/empty Any proto to trigger
    # a deserialization error.
    bad_response = computation_delegation_pb2.ComputationResponse(
        result=any_pb2.Any()
    )
    stub = self._make_stub(response=bad_response)

    with mock.patch.object(
        execution_context_helper,
        "create_computation_request",
        return_value=computation_delegation_pb2.ComputationRequest(),
    ):
      ctx = execution_context_helper.RunnerAsyncContext(stub)
      with self.assertRaisesRegex(
          RuntimeError, "Error decoding computation runner response"
      ):
        await ctx.invoke(comp)


if __name__ == "__main__":
  absltest.main()
