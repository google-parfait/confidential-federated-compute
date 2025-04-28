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
from containers.program_executor_tee.program_context import release_manager
from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc
import federated_language
import grpc
import numpy as np
import portpicker
import tensorflow_federated as tff


class ReleaseManagerTest(unittest.IsolatedAsyncioTestCase):

  async def test_release(self):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    mock_servicer = fake_data_read_write_servicer.FakeDataReadWriteServicer()
    data_read_write_pb2_grpc.add_DataReadWriteServicer_to_server(
        mock_servicer, server
    )
    port = portpicker.pick_unused_port()
    server.add_insecure_port("[::]:{}".format(port))
    server.start()

    manager = release_manager.ReleaseManager(port)
    result_uri = "my_result"
    value = 5
    await manager.release(value, result_uri)

    serialized_value, _ = tff.framework.serialize_value(
        value, federated_language.TensorType(np.int32)
    )
    expected_request = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id=result_uri.encode()
            )
        ),
        commit=True,
        data=serialized_value.SerializeToString(),
    )
    self.assertEquals(mock_servicer.get_write_call_args(), [[expected_request]])
    server.stop(grace=None)


if __name__ == "__main__":
  absltest.main()
