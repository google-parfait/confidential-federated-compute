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
import secrets
import subprocess
import sys
from typing import Optional

from fcp.protos.confidentialcompute import computation_delegation_pb2
from fcp.protos.confidentialcompute import computation_delegation_pb2_grpc
from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc
from fcp.protos.confidentialcompute import file_info_pb2
from fcp.protos.confidentialcompute import tff_config_pb2
import federated_language
from google.protobuf import any_pb2
from google.protobuf.message import DecodeError
import grpc
import portpicker
from program_executor_tee.program_context import replace_data_pointers
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


TENSORFLOW_COMPUTATION_RUNNER_BINARY_PATH = "program_executor_tee/program_context/cc/computation_runner_binary_tensorflow"
XLA_COMPUTATION_RUNNER_BINARY_PATH = (
    "program_executor_tee/program_context/cc/computation_runner_binary_xla"
)

# The default gRPC message size is 4 KiB. Increase it to 100 KiB.
_MAX_GPRC_MESSAGE_SIZE = 100 * 1024 * 1024


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
      serialized_reference_values: str = "",
      parse_read_response_fn: Callable[
          [data_read_write_pb2.ReadResponse, str, str], executor_pb2.Value
      ] = None,
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
      serialized_reference_values: A string containing a serialized
        oak.attestation.v1.ReferenceValues proto that contains reference values
        of the program worker binary to set up the client noise sessions. Need
        to be set to a non-empty string if a non-empty list of worker bns
        addresses is provided and the program workers are running on AMD SEV/SNP
        machines.
      parse_read_response_fn: A function that takes a
        data_read_write_pb2.ReadResponse, nonce, and key (from a FileInfo Data
        pointer) and returns a tff Value proto.
    """
    cache_decorator = functools.lru_cache()
    self._compiler_fn = cache_decorator(compiler_fn)
    self._uri_to_value_cache = dict()
    self._parse_read_response_fn = parse_read_response_fn

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
        f"--serialized_reference_values={serialized_reference_values}",
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
    self._data_read_write_stub = data_read_write_pb2_grpc.DataReadWriteStub(
        grpc.insecure_channel(outgoing_server_address)
    )

  def resolve_fileinfo_to_tff_value(
      self, file_info: file_info_pb2.FileInfo
  ) -> executor_pb2.Value:
    """Helper function for mapping a FileInfo Data pointer to a tff Value.

    We maintain a FileInfo uri -> tff Value cache so that external requests to
    resolve a uri only need to be made once per program instance.

    If a uri is not in the cache, a ReadRequest is sent to the DataReadWrite
    service to obtain a ReadResponse for the FileInfo uri. Next, the
    ReadResponse, the nonce used in the ReadRequest, and the FileInfo key are
    sent as args to the parse_read_response_fn provided at construction time to
    obtain the tff Value proto. In production, the provided
    parse_read_response_fn will call back to cpp code in order to decrypt the
    ReadResponse.
    """
    if file_info.uri in self._uri_to_value_cache:
      return self._uri_to_value_cache[file_info.uri]

    # Generate a nonce of 16 bytes to include in the ReadRequest. The
    # _parse_read_response_fn will later check that the received ReadResponse
    # is cryptographically tied to the same nonce.
    nonce = secrets.token_bytes(16)
    read_request = data_read_write_pb2.ReadRequest(
        uri=file_info.uri, nonce=nonce
    )

    # If there is a large amount of data, it may be split over multiple
    # ReadResponse messages. Here we combine all of the received ReadResponse
    # messages into one. Using a bytearray to construct the combined data helps
    # reduce the number of copies.
    combined_read_response = data_read_write_pb2.ReadResponse()
    combined_data = bytearray(b"")
    for read_response in self._data_read_write_stub.Read(read_request):
      if read_response.HasField("first_response_metadata"):
        combined_read_response.first_response_metadata.CopyFrom(
            read_response.first_response_metadata
        )
      combined_data.extend(read_response.data)
      if read_response.finish_read:
        combined_read_response.data = bytes(combined_data)
        combined_read_response.finish_read = True

    # Use the provided parsing function to convert the combined ReadResponse
    # message into a tff Value.
    value = self._parse_read_response_fn(
        combined_read_response, nonce, file_info.key
    )
    self._uri_to_value_cache[file_info.uri] = value
    return value

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
    if self._compiler_fn is not None:
      comp = self._compiler_fn(comp)

    session_config = tff_config_pb2.TffSessionConfig()
    serialized_comp, _ = tff.framework.serialize_value(comp)
    session_config.function.CopyFrom(serialized_comp)

    serialized_arg = None
    clients_cardinality = 0
    # TODO: For now we are assuming that the argument does not contain any
    # data pointers. This restriction will be lifted in a follow-up cl.
    if arg is not None:
      clients_cardinality = federated_language.framework.infer_cardinalities(
          arg, comp.type_signature.parameter
      )[federated_language.CLIENTS]
      serialized_arg, _ = tff.framework.serialize_value(
          arg, comp.type_signature.parameter
      )
      session_config.initial_arg.CopyFrom(
          replace_data_pointers.replace_datas(
              serialized_arg, self.resolve_fileinfo_to_tff_value
          )
      )
    session_config.num_clients = clients_cardinality

    # Send execution request for comp(arg) to the computation runner, then
    # deserialize and return the result.
    try:
      any_proto = any_pb2.Any()
      any_proto.Pack(session_config)
      delegation_request = computation_delegation_pb2.ComputationRequest(
          computation=any_proto
      )
      delegation_response = self._computation_runner_stub.Execute(
          delegation_request
      )
      result = executor_pb2.Value()
      delegation_response.result.Unpack(result)
      deserialized_result, _ = tff.framework.deserialize_value(result)
      deserialized_result = federated_language.framework.type_to_py_container(
          deserialized_result, comp.type_signature.result
      )
      return deserialized_result
    except grpc.RpcError as e:
      raise RuntimeError(
          f"Request to computation runner failed with error: {e.details()}"
      )
    except DecodeError:
      raise RuntimeError(
          "Error decoding computation runner response to tff Value"
      )
