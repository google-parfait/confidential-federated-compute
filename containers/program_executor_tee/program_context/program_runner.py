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

from containers.program_executor_tee.program_context import compilers
from containers.program_executor_tee.program_context import execution_context
from containers.program_executor_tee.program_context import release_manager
from containers.program_executor_tee.program_context.cc import computation_runner_bindings
import federated_language

# The name of the function in the customer-provided python code that wraps the
# federated program to execute.
TRUSTED_PROGRAM_KEY = "trusted_program"


async def run_program(program: str, port: int):
  """Executes a federated program.

  Args:
    program: A string that represents python code and contains a function named
      TRUSTED_PROGRAM_KEY that describes the federated program to execute. The
      TRUSTED_PROGRAM_KEY function should expect a ReleaseManager arg.
    port: The port where the DataReadWrite service is running.

  Raises:
    ValueError: If the provided python code doesn't contain TRUSTED_PROGRAM_KEY.
  """
  # TODO: Allow worker bns addresses to be set and use different compilation
  # logic in that case.
  worker_bns = []
  runner = computation_runner_bindings.ComputationRunner(worker_bns)
  federated_language.framework.set_default_context(
      execution_context.TrustedAsyncContext(
          compilers.compile_tf_to_call_dominant, runner.invoke_comp
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

  # TODO: Add additional args to the trusted_program call to allow data uris to
  # be resolved and models/checkpoints to be loaded.
  initialized_release_manager = release_manager.ReleaseManager(port)
  await trusted_program(initialized_release_manager)
