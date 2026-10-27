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
"""Helpers for executing computations with an in-process ComputationRunner."""

import asyncio
import collections
from typing import Awaitable, Callable, Optional, Sequence

import federated_language
from program_executor_tee.program_context import value_conversion
import tensorflow_federated as tff


def execute_in_process(
    runner,
    comp: object,
    arg: Optional[object] = None,
    target_worker_bns: str = "",
    return_type: Optional[federated_language.Type] = None,
) -> object:
  """Executes `comp(arg)` with a `ComputationRunner` and returns the result.

  The argument and result are passed to and from the runner as leaves (see
  `value_conversion`), so their total size is not limited by the 2 GiB proto
  limit. Tensors of bool and numeric dtypes are passed as NumPy arrays, with no
  serialization, so they are not limited by it either.

  Args:
    runner: A `ComputationRunner` (see `computation_runner_bindings.h`).
    comp: The computation to execute.
    arg: The argument of `comp`, or `None` if it has no parameter.
    target_worker_bns: The worker to execute `comp` on, if any. See
      `ComputationRunner::Execute`.
    return_type: The type of the result, used to restore its Python containers.
      Defaults to the result type of `comp`.

  Returns:
    The result of `comp(arg)`.
  """
  parameter_type = comp.type_signature.parameter
  num_clients = 0
  if arg is not None:
    cardinalities = federated_language.framework.infer_cardinalities(
        arg, parameter_type
    )
    num_clients = cardinalities.get(federated_language.CLIENTS, 0)
  comp_pb, _ = tff.framework.serialize_value(comp)
  arg_leaves = (
      None
      if arg is None
      else value_conversion.encode_value(arg, parameter_type)
  )
  result_leaves = runner.execute(
      comp_pb.SerializeToString(), arg_leaves, num_clients, target_worker_bns
  )
  if return_type is None:
    return_type = comp.type_signature.result
  return value_conversion.decode_value(result_leaves, return_type)


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
  """AsyncContext that executes computations with a `ComputationRunner`.

  Used by MergeableCompExecutionContext. Computations whose type signature
  contains @CLIENTS placement are executed on the configured worker
  (`worker_bns`). All other computations are executed locally.
  """

  def __init__(self, runner, worker_bns: str = ""):
    self._runner = runner
    self.worker_bns = worker_bns

  async def invoke(
      self,
      comp: federated_language.framework.Computation,
      arg: Optional[object] = None,
  ) -> object:
    target_worker_bns = (
        self.worker_bns
        if contains_clients_placement(comp.type_signature)
        else ""
    )
    loop = asyncio.get_running_loop()
    try:
      return await loop.run_in_executor(
          None, execute_in_process, self._runner, comp, arg, target_worker_bns
      )
    except Exception as e:
      raise RuntimeError(
          f"Request to computation runner failed with error: {e}"
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
      f" and {len(execution_contexts)} contexts with background merge."
      f" max_retries_per_subround: {max_retries_per_subround}."
  )
  pending_tasks = {}  # task -> (context, subround_idx, subround_arg)
  subround_retries = collections.defaultdict(int)
  context_handling_counts = collections.defaultdict(int)
  accumulated_result = initial_result
  last_ctx = None
  last_exception = None
  done = set()

  # Asynchronous background queue for postprocessing/merge.
  # This decouples worker subround dispatch from postprocessing execution:
  # as soon as a worker subround task finishes, its context is immediately
  # returned to `available_contexts` and re-dispatched to the next subround,
  # without waiting for `postprocessing` (e.g., merge on root) to finish.
  merge_queue = asyncio.Queue()
  sentinel = object()

  async def _postprocessing_consumer():
    nonlocal accumulated_result, last_ctx
    while True:
      item = await merge_queue.get()
      if item is sentinel:
        merge_queue.task_done()
        break
      partial_result, ctx = item
      try:
        last_ctx = ctx
        accumulated_result = await postprocessing(
            accumulated_result, partial_result, ctx
        )
      finally:
        merge_queue.task_done()

  consumer_task = asyncio.create_task(_postprocessing_consumer())

  def _cleanup_tasks():
    consumer_task.cancel()
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
        _cleanup_tasks()
        raise RuntimeError(
            "All execution contexts failed: no available worker contexts"
            f" remaining to execute subrounds. Last error: {last_exception}"
        )
      break

    # If the background consumer encountered an exception, fail fast.
    if consumer_task.done():
      _cleanup_tasks()
      consumer_task.result()

    # Wait for the first subround task(s) to complete.
    done, _ = await asyncio.wait(
        pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
    )

    for done_task in done:
      ctx, subround_idx, subround_arg = pending_tasks.pop(done_task)
      try:
        partial_result = done_task.result()
        # Immediately mark the worker context available for the next subround
        # without waiting for postprocessing to finish.
        available_contexts.add(ctx)
        context_handling_counts[ctx] += 1
        # Offload postprocessing/merge onto the background consumer.
        # Note: `ctx` is passed to satisfy the TFF `postprocessing` / `_merge_results`
        # callback signature. When `ctx.invoke(comp.merge, ...)` is invoked,
        # `RunnerAsyncContext` checks `contains_clients_placement(comp.type_signature)`.
        # Since `merge` operates purely on server/accumulator state and has no
        # CLIENTS placement, `target_worker_bns` is empty and the merge executes
        # locally on the root container without delegating to or blocking any remote worker.
        merge_queue.put_nowait((partial_result, ctx))
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

  # Signal the background consumer to finish and wait for all merges to complete.
  await merge_queue.put(sentinel)
  await consumer_task

  print(
      f"run_resilient_subrounds: completed. Context handling counts:"
      f" {[context_handling_counts[ctx] for ctx in execution_contexts]}"
  )
  return accumulated_result, last_ctx

