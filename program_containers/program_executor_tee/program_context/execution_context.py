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
import os
import subprocess
import sys
from typing import Optional

from fcp.protos.confidentialcompute import computation_delegation_pb2_grpc
import federated_language
from google.protobuf.message import DecodeError
import grpc
import portpicker
from program_executor_tee.program_context import execution_context_helper

# Increase gRPC message size limit to 2GB
_MAX_GPRC_MESSAGE_SIZE = 2 * 1000 * 1000 * 1000


class TrustedContext(federated_language.program.FederatedContext):
  """Execution context for executing computations in an Federated Program."""

  def __init__(
      self,
      compiler_fn: Optional[
          Callable[
              [federated_language.framework.ConcreteComputation],
              federated_language.framework.ConcreteComputation,
          ]
      ],
      computation_runner_binary_path: str,
      outgoing_server_address: str,
      worker_bns: list[str] = [],
      serialized_reference_values: bytes = b"",
      max_concurrent_computation_calls=-1,
      use_elastic_composing_executor: bool = False,
  ):
    """Initializes the execution context with an invoke helper function.

    Args:
      compiler_fn: Python function that will be used to compile a computation.
      computation_runner_binary_path: The local path to the computation runner
        binary.
      outgoing_server_address: The address at which the untrusted root server
        can be reached for data read/write requests and computation delegation
        requests.
      worker_bns: A list of worker bns addresses.
      serialized_reference_values: A bytestring containing a serialized
        oak.attestation.v1.ReferenceValues proto that contains reference values
        of the program worker binary to set up the client noise sessions. Need
        to be set to a non-empty string if a non-empty list of worker bns
        addresses is provided and the program workers are running on AMD SEV/SNP
        machines.
      max_concurrent_computation_calls: The number of calls that can be handled
        in parallel by the leaf executors. Non-positive values indicate no
        maximum.
      use_elastic_composing_executor: Whether to use ElasticComposingExecutor
        instead of ComposingExecutor.
    """

    if compiler_fn is not None:
      cache_decorator = functools.lru_cache()
      self._compiler_fn = cache_decorator(compiler_fn)
    else:
      self._compiler_fn = None

    # Start the computation runner on a different process.
    computation_runner_binary_path = os.path.join(
        os.getcwd(),
        computation_runner_binary_path,
    )
    if not os.path.isfile(computation_runner_binary_path):
      raise RuntimeError(
          f"Expected a worker binary at {computation_runner_binary_path}."
      )
    computation_runner_port = portpicker.pick_unused_port()
    args = [
        computation_runner_binary_path,
        f"--computatation_runner_port={computation_runner_port}",
        f"--outgoing_server_address={outgoing_server_address}",
        f"--worker_bns={','.join(worker_bns)}",
        # Convert the base64-encoded serialized reference value bytestring to a string.
        # The computation runner will decode this arg into a ReferenceValues proto.
        f"--serialized_reference_values={serialized_reference_values.decode('utf-8')}",
        f"--max_concurrent_computation_calls={max_concurrent_computation_calls}",
        f"--use_elastic_composing_executor={use_elastic_composing_executor}",
    ]
    self._process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    channel_options = [
        ("grpc.max_send_message_length", _MAX_GPRC_MESSAGE_SIZE),
        ("grpc.max_receive_message_length", _MAX_GPRC_MESSAGE_SIZE),
    ]
    channel = grpc.insecure_channel(
        "[::1]:{}".format(computation_runner_port), options=channel_options
    )
    grpc.channel_ready_future(channel).result(timeout=5)
    self._computation_runner_stub = (
        computation_delegation_pb2_grpc.ComputationDelegationStub(channel)
    )

  def close(self):
    if self._process is not None:
      self._process.terminate()
      self._process.wait()
      self._process = None

  def __del__(self):
    self.close()

  def invoke(self, comp: object, arg: Optional[object]) -> object:
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
    # Save the computation return type since the compile call below may drop
    # some of the information we need later.
    comp_return_type = comp.type_signature.result

    if self._compiler_fn is not None:
      comp = self._compiler_fn(comp)

    delegation_request = execution_context_helper.create_computation_request(
        comp, arg
    )
    try:
      delegation_response = self._computation_runner_stub.Execute(
          delegation_request
      )
      return (
          execution_context_helper.unpack_and_deserialize_computation_response(
              delegation_response, comp_return_type
          )
      )
    except grpc.RpcError as e:
      raise RuntimeError(
          f"Request to computation runner failed with error: {e.details()}"
      )
    except DecodeError:
      raise RuntimeError(
          "Error decoding computation runner response to tff Value"
      )
