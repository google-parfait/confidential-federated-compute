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

import base64
from collections.abc import Callable, Sequence
import inspect
from typing import Optional

from fcp.confidentialcompute.python import program_input_provider
from fcp.confidentialcompute.python import external_service_handle
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2

# The name of the function in the customer-provided python code that wraps the
# federated program to execute.
TRUSTED_PROGRAM_KEY = "trusted_program"


def run_program(
    initialize_fn: Optional[Callable[[], None]],
    program: bytes,
    client_ids: list[str],
    client_data_directory: str,
    model_id_to_zip_file: dict[str, str],
    outgoing_server_address: str,
    resolve_uri_to_tensor_fn: Callable[[str, str], tensor_pb2.TensorProto],
    release_unencrypted_fn: Callable[[bytes, str], None],
    save_recovery_info_fn: Callable[
        [bytes, str, Sequence[tuple[bytes, str]]], None
    ],
    restore_recovery_info_fn: Callable[[str], bytes],
):
  """Executes a federated program.

  Args:
    initialize_fn: An optional initialization function to call prior to
      executing the program.
    program: A base64-encoded bytestring that represents python code and
      contains a function named TRUSTED_PROGRAM_KEY that describes the federated
      program to execute. The TRUSTED_PROGRAM_KEY function should expect a
      ReleaseManager arg.
    client_ids: A list of strings representing the client upload base filenames
      that can be used during program execution. The concatenation of
      'client_data_directory' and a client id is the full path to a client file.
    client_data_directory: The directory containing the client data.
    model_id_to_zip_file: A dictionary mapping model ids to the paths of the zip
      files containing the model weights for those models.
    outgoing_server_address: The address at which the untrusted root server can
      be reached for data read/write requests and computation delegation
      requests.
    resolve_uri_to_tensor_fn: A function that resolves pointers to data.
      Expects two args (the uri and the key) and returns the resolved tensor.
    release_unencrypted_fn: A function that releases unencrypted values to the
      external service. Expects two args (the data and the key).
    save_recovery_info_fn: A function that saves recovery information. Expects
      three args (the recovery info, the recovery key, and a list of value and
      key pairs to release as unencrypted data).
    restore_recovery_info_fn: A function that restores recovery information.
      Expects one arg (the key) and returns the recovery info.

  Raises:
    ValueError: If the provided python code doesn't contain TRUSTED_PROGRAM_KEY.
  """
  if initialize_fn is not None:
    initialize_fn()

  # Decode the program, which must be provided as a base64-encoded bytestring.
  try:
    program = base64.b64decode(program).decode("utf-8")
  except:
    raise ValueError("The provided program must be base64-encoded.")

  # Load the provided python code into a namespace and extract the function
  # wrapping the program to run.
  program_namespace = {}
  exec(program, program_namespace)
  if TRUSTED_PROGRAM_KEY not in program_namespace:
    raise ValueError(
        "The provided program must have a " + TRUSTED_PROGRAM_KEY + " function."
    )
  trusted_program = program_namespace[TRUSTED_PROGRAM_KEY]

  initialized_external_service_handle = (
      external_service_handle.ExternalServiceHandle(
          outgoing_server_address,
          client_ids,
          client_data_directory,
          model_id_to_zip_file,
          resolve_uri_to_tensor_fn = resolve_uri_to_tensor_fn,
          release_unencrypted_fn = release_unencrypted_fn,
          save_recovery_info_fn = save_recovery_info_fn,
          restore_recovery_info_fn = restore_recovery_info_fn,
      )
  )

  # Dispatch based on the trusted_program's signature: 1-arg (new) programs
  # receive only the ExternalServiceHandle, while 2-arg (legacy) programs
  # receive both a ProgramInputProvider and an ExternalServiceHandle.
  sig = inspect.signature(trusted_program)
  num_params = len(sig.parameters)
  if num_params == 1:
    trusted_program(initialized_external_service_handle)
  elif num_params == 2:
    # TODO(b/497752916): Remove this legacy 2-arg dispatch path once all
    # customer programs have been migrated to the 1-arg signature.
    input_provider = program_input_provider.ProgramInputProvider(
        client_ids,
        client_data_directory,
        model_id_to_zip_file,
        resolve_uri_to_tensor_fn,
    )
    trusted_program(input_provider, initialized_external_service_handle)
  else:
    raise ValueError(
        f"trusted_program must accept 1 or 2 arguments, got {num_params}."
    )
