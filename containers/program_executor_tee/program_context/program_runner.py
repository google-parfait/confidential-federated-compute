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
from containers.program_executor_tee.program_context import compilers
from containers.program_executor_tee.program_context import execution_context
from containers.program_executor_tee.program_context import release_manager
from fcp.confidentialcompute.python import program_input_provider
from fcp.protos.confidentialcompute import data_read_write_pb2
import federated_language
from tensorflow_federated.proto.v0 import executor_pb2

# The name of the function in the customer-provided python code that wraps the
# federated program to execute.
TRUSTED_PROGRAM_KEY = "trusted_program"


async def run_program(
    program: str,
    client_ids: list[str],
    client_data_directory: str,
    model_id_to_zip_file: dict[str, str],
    outgoing_server_address: str,
    worker_bns: list[str] = [],
    attester_id: str = "",
    parse_read_response_fn: Callable[
        [data_read_write_pb2.ReadResponse, str, str], executor_pb2.Value
    ] = None,
):
  """Executes a federated program.

  Args:
    program: A string that represents python code and contains a function named
      TRUSTED_PROGRAM_KEY that describes the federated program to execute. The
      TRUSTED_PROGRAM_KEY function should expect a ReleaseManager arg.
    client_ids: A list of strings representing the client upload base filenames
      that can be used during program execution. The concatenation of
      'client_data_directory' and a client id is the full path to a client file.
    client_data_directory: The directory containing the client data.
    model_id_to_zip_file: A dictionary mapping model ids to the paths of the zip
      files containing the model weights for those models.
    outgoing_server_address: The address at which the untrusted root server can
      be reached for data read/write requests and computation delegation
      requests.
    worker_bns: A list of worker bns addresses.
    attester_id: The attester id for setting up the noise sessions used for
      distributed execution. Needs to be set to a non-empty string if a
      non-empty list of worker bns addresses is provided.
    parse_read_response_fn: A function that takes a
      data_read_write_pb2.ReadResponse, nonce, and key (from a FileInfo Data
      pointer) and returns a tff Value proto.

  Raises:
    ValueError: If the provided python code doesn't contain TRUSTED_PROGRAM_KEY.
  """
  federated_language.framework.set_default_context(
      execution_context.TrustedAsyncContext(
          compilers.compile_tf_to_call_dominant,
          outgoing_server_address,
          worker_bns,
          attester_id,
          parse_read_response_fn,
      )
  )

  # Load the provided python code into a namespace and extract the function
  # wrapping the program to run.
  program_namespace = {}
  exec(program, program_namespace)
  if TRUSTED_PROGRAM_KEY not in program_namespace:
    raise ValueError(
        "The provided program must have a " + TRUSTED_PROGRAM_KEY + " function."
    )
  trusted_program = program_namespace[TRUSTED_PROGRAM_KEY]

  input_provider = program_input_provider.ProgramInputProvider(
      client_ids, client_data_directory, model_id_to_zip_file
  )
  initialized_release_manager = release_manager.ReleaseManager(
      outgoing_server_address
  )
  await trusted_program(input_provider, initialized_release_manager)
