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

from concurrent import futures
import unittest

from absl.testing import absltest
from containers.program_executor_tee.program_context import fake_data_read_write_servicer
from containers.program_executor_tee.program_context import program_runner
from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc
import federated_language
import grpc
import numpy as np
import portpicker
import tensorflow_federated as tff


class ProgramRunnerIntegrationTest(unittest.IsolatedAsyncioTestCase):

  async def test_run_program(self):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    mock_servicer = fake_data_read_write_servicer.FakeDataReadWriteServicer()
    data_read_write_pb2_grpc.add_DataReadWriteServicer_to_server(
        mock_servicer, server
    )
    port = portpicker.pick_unused_port()
    server.add_insecure_port("[::]:{}".format(port))
    server.start()

    program_string = """
import federated_language
import numpy as np

async def trusted_program(release_manager):
    client_data_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(
        [client_data_type, server_state_type]
    )
    def my_comp(client_data, server_state):
        return federated_language.federated_sum(client_data), server_state

    result1, result2 = await my_comp([1, 2], 10)

    await release_manager.release(result1, "result1")
    await release_manager.release(result2, "result2")
    """

    await program_runner.run_program(program_string, port)

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
    value_10, _ = tff.framework.serialize_value(
        10, federated_language.TensorType(np.int32)
    )
    expected_request_2 = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id="result2".encode()
            )
        ),
        commit=True,
        data=value_10.SerializeToString(),
    )

    self.assertEquals(
        mock_servicer.get_write_call_args(),
        [[expected_request_1], [expected_request_2]],
    )

    server.stop(grace=None)

  async def test_run_program_without_trusted_program_function(self):
    program_string = """
def incorrectly_named_trusted_program(release_manager):
    return 10
    """

    with self.assertRaises(ValueError) as context:
      await program_runner.run_program(
          program_string, portpicker.pick_unused_port()
      )
    self.assertEqual(
        str(context.exception),
        "The provided program must have a "
        + program_runner.TRUSTED_PROGRAM_KEY
        + " function.",
    )


if __name__ == "__main__":
  absltest.main()
