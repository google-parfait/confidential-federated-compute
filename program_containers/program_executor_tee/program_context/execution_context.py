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

from collections.abc import Callable
import functools
from typing import Optional

import federated_language
from program_executor_tee.program_context import execution_context_helper
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context


class TrustedContext(federated_language.program.FederatedContext):
  """Execution context for executing computations in an Federated Program."""

  def __init__(
      self,
      compiler_fn: Optional[Callable],
      computation_runner_cls: Callable,
      mergeable_comp_compiler_fn: Optional[Callable] = None,
      outgoing_server_address: str = "",
      worker_bns: list[str] = [],
      serialized_reference_values: bytes = b"",
      max_concurrent_computation_calls: int = -1,
      use_elastic_composing_executor: bool = False,
      num_subrounds: Optional[int] = None,
  ):
    """Initializes the execution context with an invoke helper function.

    Args:
      compiler_fn: Python function that will be used to compile a computation
        for direct execution (e.g. to call-dominant form).
      computation_runner_cls: The `ComputationRunner` class of the bindings
        module for the leaf executor backend, such as
        `computation_runner_bindings_jax.ComputationRunner`.
      mergeable_comp_compiler_fn: Optional Python function that will be used to
        compile a computation to MergeableCompForm for mergeable execution. When
        provided and when the computation can be compiled to MergeableCompForm,
        then MergeableCompExecutionContext (configured with resilient subround
        failover) will be used.
      outgoing_server_address: The address at which the untrusted root server
        can be reached for data read/write requests and computation delegation
        requests.
      worker_bns: A list of worker bns addresses.
      serialized_reference_values: A base64-encoded bytestring containing a
        serialized oak.attestation.v1.ReferenceValues proto that contains
        reference values of the program worker binary to set up the client noise
        sessions. Need to be set to a non-empty string if a non-empty list of
        worker bns addresses is provided and the program workers are running on
        AMD SEV/SNP machines.
      max_concurrent_computation_calls: The number of calls that can be handled
        in parallel by the leaf executors. Non-positive values indicate no
        maximum.
      use_elastic_composing_executor: Whether to use ElasticComposingExecutor
        instead of ComposingExecutor.
      num_subrounds: Optional number of subrounds for the
        MergeableCompExecutionContext. If not set, defaults to the number of
        worker contexts.
    """
    if (
        use_elastic_composing_executor
        and mergeable_comp_compiler_fn is not None
    ):
      raise ValueError(
          "Only one of use_elastic_composing_executor or"
          " mergeable_comp_compiler_fn can be set."
      )

    if compiler_fn is not None:
      cache_decorator = functools.lru_cache()
      self._compiler_fn = cache_decorator(compiler_fn)
    else:
      self._compiler_fn = None

    if mergeable_comp_compiler_fn is not None:
      cache_decorator = functools.lru_cache()
      self._mergeable_comp_compiler_fn = cache_decorator(
          mergeable_comp_compiler_fn
      )
    else:
      self._mergeable_comp_compiler_fn = None

    self._worker_bns = worker_bns
    self._use_elastic_composing_executor = use_elastic_composing_executor

    self._runner = computation_runner_cls(
        worker_bns=worker_bns,
        serialized_reference_values=serialized_reference_values,
        outgoing_server_address=outgoing_server_address,
        use_elastic_composing_executor=use_elastic_composing_executor,
        use_mergeable_execution_context=(
            mergeable_comp_compiler_fn is not None
        ),
        max_concurrent_computation_calls=max_concurrent_computation_calls,
    )

    if self._mergeable_comp_compiler_fn is not None and self._worker_bns:
      worker_contexts = [
          execution_context_helper.RunnerAsyncContext(self._runner, bns)
          for bns in self._worker_bns
      ]
      self._mergeable_context = (
          mergeable_comp_execution_context.MergeableCompExecutionContext(
              async_contexts=worker_contexts,
              num_subrounds=num_subrounds,
              pool_runner=execution_context_helper.run_resilient_subrounds,
          )
      )
    else:
      self._mergeable_context = None

  def _invoke_unpartitioned(
      self, comp: object, arg: Optional[object]
  ) -> object:
    comp_return_type = comp.type_signature.result

    if self._compiler_fn is not None:
      comp = self._compiler_fn(comp)

    return execution_context_helper.execute_in_process(
        self._runner, comp, arg, return_type=comp_return_type
    )

  def invoke(self, comp: object, arg: Optional[object]) -> object:
    """Executes comp(arg).

    Execution follows this decision and fallback order:
    1. If `mergeable_comp_compiler_fn` is provided and worker BNS addresses are
       configured, attempts to compile the computation into `MergeableCompForm`.
       If compilation succeeds, delegates execution to
       `MergeableCompExecutionContext`, which distributes `up_to_merge`
       work across workers (with dynamic failover if any workers become
       unavailable) and executes `merge`/`after_merge` on the local server stack.
    2. If `mergeable_comp_compiler_fn` is not provided, if worker BNS addresses
       are not configured, or if compilation to `MergeableCompForm` raises an
       exception (e.g. for non-mergeable or pure server computations), falls
       back to `_invoke_unpartitioned`. This compiles the computation using
       `compiler_fn` and executes it directly with the in-process
       ComputationRunner.

    Args:
      comp: The deserialized comp.
      arg: The deserialized arg.

    Returns:
      A deserialized result.
    """
    # Only partition computations across workers if they take client-placed
    # inputs to partition into subrounds. Pure server computations (e.g.
    # initialize() with no arguments or computations with only server-placed
    # parameters) execute directly on the root container via _invoke_unpartitioned.
    has_clients_parameter = (
        comp.type_signature.parameter is not None
        and execution_context_helper.contains_clients_placement(
            comp.type_signature.parameter
        )
    )

    if (
        self._mergeable_context is not None
        and self._worker_bns
        and has_clients_parameter
    ):
      try:
        compiled_comp = self._mergeable_comp_compiler_fn(comp)
      except Exception as e:
        print(
            f"Compiling computation to MergeableCompForm failed: {e}."
            " Falling back to _invoke_unpartitioned."
        )
        compiled_comp = None

      if compiled_comp is not None:
        print("Executing via MergeableCompExecutionContext.")
        return self._mergeable_context.invoke(compiled_comp, arg)

    print("Executing via _invoke_unpartitioned.")
    return self._invoke_unpartitioned(comp, arg)
