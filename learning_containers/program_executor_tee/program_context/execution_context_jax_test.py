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

import unittest

from absl.testing import absltest
import federated_language
from federated_language_jax.computation import jax_computation
import portpicker
from program_executor_tee.program_context import execution_context
from program_executor_tee.program_context import test_helpers
from program_executor_tee.program_context.cc import fake_service_bindings_jax


XLA_COMPUTATION_RUNNER_BINARY_PATH = (
    "program_executor_tee/program_context/cc/computation_runner_binary_xla"
)


class ExecutionContextTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(
        self.untrusted_root_port, "Failed to pick an unused port."
    )
    self.outgoing_server_address = f"[::1]:{self.untrusted_root_port}"
    self.worker_bns = []
    self.serialized_reference_values = b""
    self.data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    self.server = fake_service_bindings_jax.FakeServer(
        self.untrusted_root_port, self.data_read_write_service, None
    )
    self.server.start()

  def tearDown(self):
    self.server.stop()

  async def test_execution_context_jax_computation(self):
    context = execution_context.TrustedContext(
        lambda x: x,
        XLA_COMPUTATION_RUNNER_BINARY_PATH,
        self.outgoing_server_address,
        self.worker_bns,
        self.serialized_reference_values,
        test_helpers.parse_read_response_fn,
    )

    with federated_language.framework.get_context_stack().install(context):

      @jax_computation.jax_computation
      def comp():
        return 10

      result = comp()

    self.assertEqual(result, 10)


if __name__ == "__main__":
  absltest.main()
