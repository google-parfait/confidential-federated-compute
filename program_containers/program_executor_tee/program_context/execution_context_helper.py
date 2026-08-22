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
"""Helper utilities for serialized computation requests and responses."""

import asyncio
import collections
from typing import Awaitable, Callable, Optional, Sequence

from fcp.protos.confidentialcompute import computation_delegation_pb2
from fcp.protos.confidentialcompute import computation_delegation_pb2_grpc
from fcp.protos.confidentialcompute import tff_config_pb2
import federated_language
from google.protobuf import any_pb2
import grpc
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


def create_computation_request(
    comp: object, arg: Optional[object] = None
) -> computation_delegation_pb2.ComputationRequest:
  """Builds a ComputationRequest containing the serialized comp, arg, and num_clients."""
  session_config = tff_config_pb2.TffSessionConfig()
  serialized_comp, _ = tff.framework.serialize_value(comp)
  session_config.function.CopyFrom(serialized_comp)

  serialized_arg = None
  clients_cardinality = 0
  if arg is not None:
    cardinalities = federated_language.framework.infer_cardinalities(
        arg, comp.type_signature.parameter
    )
    clients_cardinality = cardinalities.get(
        federated_language.CLIENTS, clients_cardinality
    )
    serialized_arg, _ = tff.framework.serialize_value(
        arg, comp.type_signature.parameter
    )
    session_config.initial_arg.CopyFrom(serialized_arg)
  session_config.num_clients = clients_cardinality

  any_proto = any_pb2.Any()
  any_proto.Pack(session_config)
  return computation_delegation_pb2.ComputationRequest(computation=any_proto)


def unpack_and_deserialize_computation_response(
    response: computation_delegation_pb2.ComputationResponse,
    return_type: federated_language.Type,
) -> object:
  """Unpacks and deserializes the ComputationResponse into a Python structure."""
  result_value = executor_pb2.Value()
  response.result.Unpack(result_value)
  deserialized_result, _ = tff.framework.deserialize_value(result_value)
  if isinstance(
      deserialized_result, federated_language.common_libs.structure.Struct
  ):
    deserialized_result = (
        federated_language.common_libs.structure.to_odict_or_tuple(
            deserialized_result
        )
    )
  return federated_language.framework.to_structure_with_type(
      deserialized_result, return_type
  )


def contains_clients_placement(type_spec: federated_language.Type) -> bool:
  """Checks if a TFF Type specification contains @CLIENTS placement."""
  if isinstance(type_spec, federated_language.FederatedType):
    if type_spec.placement is federated_language.CLIENTS:
      return True
    return contains_clients_placement(type_spec.member)
  elif isinstance(type_spec, federated_language.StructType):
    return any(contains_clients_placement(elem) for elem in type_spec)
  elif isinstance(type_spec, federated_language.FunctionType):
    param_has_clients = (
        contains_clients_placement(type_spec.parameter)
        if type_spec.parameter is not None
        else False
    )
    return param_has_clients or contains_clients_placement(type_spec.result)
  elif isinstance(type_spec, federated_language.SequenceType):
    return contains_clients_placement(type_spec.element)
  return False


class RunnerAsyncContext(federated_language.framework.AsyncContext):
  """AsyncContext that delegates computations to the ComputationRunner gRPC service.

  Used by MergeableCompExecutionContext. For computations whose type
  signature contains @CLIENTS placement, the request's worker_bns is set
  to the configured worker_bns so the ComputationRunner forwards execution
  to that worker. For all other computations, worker_bns is set to empty,
  which causes the ComputationRunner to execute them locally.
  """

  def __init__(
      self,
      computation_runner_stub: computation_delegation_pb2_grpc.ComputationDelegationStub,
      worker_bns: str = "",
  ):
    self._stub = computation_runner_stub
    self.worker_bns = worker_bns

  async def invoke(
      self,
      comp: federated_language.framework.Computation,
      arg: Optional[object] = None,
  ) -> object:
    comp_return_type = comp.type_signature.result
    target_worker_bns = (
        self.worker_bns
        if contains_clients_placement(comp.type_signature)
        else ""
    )
    delegation_request = create_computation_request(comp, arg)
    delegation_request.worker_bns = target_worker_bns

    loop = asyncio.get_running_loop()
    try:
      delegation_response = await loop.run_in_executor(
          None, self._stub.Execute, delegation_request
      )
      return unpack_and_deserialize_computation_response(
          delegation_response, comp_return_type
      )
    except grpc.RpcError as e:
      details = (
          e.details()
          if hasattr(e, "details") and callable(e.details)
          else str(e)
      )
      raise RuntimeError(
          f"Request to computation runner failed with error: {details}"
          f" (target worker: '{target_worker_bns or '(local)'}')"
      ) from e
    except Exception as e:
      raise RuntimeError(
          f"Error decoding computation runner response: {e}"
          f" (target worker: '{target_worker_bns or '(local)'}')"
      ) from e


async def run_resilient_subrounds(
    task_fn: Callable[
        [object, federated_language.framework.AsyncContext], Awaitable[object]
    ],
    arg_list: Sequence[object],
    execution_contexts: Sequence[federated_language.framework.AsyncContext],
    initial_result: object = None,
    postprocessing: Optional[
        Callable[
            [object, object, federated_language.framework.AsyncContext],
            Awaitable[object],
        ]
    ] = None,
    max_retries_per_subround: int = 4,
) -> tuple[object, Optional[federated_language.framework.AsyncContext]]:
  """Runs tasks against a pool of async contexts with dynamic re-queuing and failover."""
  if postprocessing is None:

    async def postprocessing(acc, val, ctx):
      del ctx  # Unused
      return (acc or []) + [val]

  work_queue = collections.deque(enumerate(arg_list))
  available_contexts = set(execution_contexts)
  print(
      f"run_resilient_subrounds: starting with {len(arg_list)} subrounds"
      f" and {len(execution_contexts)} contexts."
      f" max_retries_per_subround: {max_retries_per_subround}."
  )
  pending_tasks = {}  # task -> (context, subround_idx, subround_arg)
  subround_retries = collections.defaultdict(int)
  context_handling_counts = collections.defaultdict(int)
  accumulated_result = initial_result
  last_ctx = None
  last_exception = None
  done = set()

  def _cleanup_tasks():
    for t in set(pending_tasks) | set(done):
      if not t.done():
        t.cancel()
      elif not t.cancelled():
        t.exception()

  while work_queue or pending_tasks:
    # Dispatch available work onto idle worker contexts.
    while work_queue and available_contexts:
      ctx = available_contexts.pop()
      subround_idx, subround_arg = work_queue.popleft()
      task = task_fn(subround_arg, ctx)
      pending_tasks[task] = (ctx, subround_idx, subround_arg)

    if not pending_tasks:
      if work_queue:
        raise RuntimeError(
            "All execution contexts failed: no available worker contexts"
            f" remaining to execute subrounds. Last error: {last_exception}"
        )
      break

    # Wait for the first subround task(s) to complete.
    done, _ = await asyncio.wait(
        pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
    )

    for done_task in done:
      ctx, subround_idx, subround_arg = pending_tasks.pop(done_task)
      try:
        partial_result = done_task.result()
        last_ctx = ctx
        available_contexts.add(ctx)
        context_handling_counts[ctx] += 1
        accumulated_result = await postprocessing(
            accumulated_result, partial_result, ctx
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        last_exception = e
        # Re-queue subround arg for retry on any exception if retry budget allows.
        subround_retries[subround_idx] += 1
        ctx_bns = getattr(ctx, "worker_bns", ctx)
        print(
            f"Subround {subround_idx} failed on context {ctx_bns}."
            f" Error: {e}."
            f" Retry {subround_retries[subround_idx]}/{max_retries_per_subround}."
            f" Available contexts:"
            f" {[getattr(c, 'worker_bns', c) for c in available_contexts]}."
            f" Pending tasks: {len(pending_tasks)},"
            f" Work queue: {len(work_queue)}."
        )
        if subround_retries[subround_idx] > max_retries_per_subround:
          _cleanup_tasks()
          raise RuntimeError(
              f"Subround {subround_idx} failed after"
              f" {max_retries_per_subround} retries."
              f" Error: {e}"
          )
        work_queue.append((subround_idx, subround_arg))

  print(
      f"run_resilient_subrounds: completed. Context handling counts:"
      f" {[context_handling_counts[ctx] for ctx in execution_contexts]}"
  )
  return accumulated_result, last_ctx
