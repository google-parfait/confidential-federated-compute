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
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


class TrustedAsyncContext(federated_language.framework.AsyncContext):
  """Asynchronous TFF execution context for executing TFF computations."""

  def __init__(
      self,
      compiler_fn: Optional[
          Callable[
              [federated_language.framework.ConcreteComputation],
              federated_language.framework.ConcreteComputation,
          ]
      ],
      invoke_fn: Callable[
          [int, executor_pb2.Value, executor_pb2.Value], executor_pb2.Value
      ],
  ):
    """Initializes the execution context with an invoke helper function.

    Args:
      compiler_fn: Python function that will be used to compile a computation.
      invoke_fn: Executes comp(arg) when called with the comp's client
        cardinality, the serialized comp, and the serialized arg. Returns a
        serialized result.
    """
    cache_decorator = functools.lru_cache()
    self._compiler_fn = cache_decorator(compiler_fn)
    self._invoke_fn = invoke_fn

  async def invoke(self, comp: object, arg: Optional[object]) -> object:
    """Executes comp(arg).

    Compiles the computation, serializes the comp and arg, and delegates
    execution to the helper function provided to the constructor, before
    returning a deserialized result.

    Args:
      comp: The deserialized comp.
      arg: The deserialized arg.

    Returns:
      A deserialized result.
    """
    if self._compiler_fn is not None:
      comp = self._compiler_fn(comp)

    serialized_comp, _ = tff.framework.serialize_value(comp)
    serialized_arg = None
    clients_cardinality = 0

    if arg is not None:
      serialized_arg, _ = tff.framework.serialize_value(
          arg, comp.type_signature.parameter
      )
      clients_cardinality = federated_language.framework.infer_cardinalities(
          arg, comp.type_signature.parameter
      )[federated_language.CLIENTS]

    result = self._invoke_fn(
        clients_cardinality,
        serialized_comp,
        serialized_arg,
    )
    deserialized_result, _ = tff.framework.deserialize_value(result)
    return deserialized_result
