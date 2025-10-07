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
import unittest

from absl.testing import absltest
from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
import federated_language
import numpy as np
import portpicker
from program_executor_tee.program_context import program_runner
from program_executor_tee.program_context.cc import fake_service_bindings_jax
import tensorflow_federated as tff


class ProgramRunnerTest(unittest.IsolatedAsyncioTestCase):

  async def test_run_program(self):
    untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(untrusted_root_port, "Failed to pick an unused port.")
    data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    server = fake_service_bindings_jax.FakeServer(
        untrusted_root_port, data_read_write_service, None
    )
    server.start()

    program_string = """
async def trusted_program(input_provider, release_manager):
    result1 = 1+2
    result2 = 1*2
    await release_manager.release(result1, "result1")
    await release_manager.release(result2, "result2")
    """

    value_3, _ = tff.framework.serialize_value(
        3, federated_language.TensorType(np.int32)
    )
    expected_request_1 = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id="result1".encode()
            )
        ),
        commit=True,
        data=value_3.SerializeToString(),
    )
    value_2, _ = tff.framework.serialize_value(
        2, federated_language.TensorType(np.int32)
    )
    expected_request_2 = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id="result2".encode()
            )
        ),
        commit=True,
        data=value_2.SerializeToString(),
    )

    await program_runner.run_program(
        initialize_fn=None,
        program=base64.b64encode(program_string.encode("utf-8")),
        client_ids=[],
        client_data_directory="",
        model_id_to_zip_file={},
        outgoing_server_address=f"[::1]:{untrusted_root_port}",
    )

    self.assertEqual(
        data_read_write_service.get_write_call_args(),
        [[expected_request_1], [expected_request_2]],
    )

    server.stop()

  async def test_run_program_without_trusted_program_function(self):
    program_string = """
def incorrectly_named_trusted_program(release_manager):
    return 10
    """

    with self.assertRaises(ValueError) as context:
      await program_runner.run_program(
          initialize_fn=None,
          program=base64.b64encode(program_string.encode("utf-8")),
          client_ids=[],
          client_data_directory="",
          model_id_to_zip_file={},
          outgoing_server_address="",
      )
    self.assertEqual(
        str(context.exception),
        "The provided program must have a "
        + program_runner.TRUSTED_PROGRAM_KEY
        + " function.",
    )


if __name__ == "__main__":
  absltest.main()
