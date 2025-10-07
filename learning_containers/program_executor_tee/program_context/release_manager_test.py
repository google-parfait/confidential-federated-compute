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
from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
import federated_language
import numpy as np
import portpicker
from program_executor_tee.program_context import release_manager
from program_executor_tee.program_context.cc import fake_service_bindings_jax
import tensorflow_federated as tff


class ReleaseManagerTest(unittest.IsolatedAsyncioTestCase):

  async def test_release(self):
    untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(untrusted_root_port, "Failed to pick an unused port.")
    data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    server = fake_service_bindings_jax.FakeServer(
        untrusted_root_port, data_read_write_service, None
    )
    server.start()

    manager = release_manager.ReleaseManager(f"[::1]:{untrusted_root_port}")
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
    self.assertEqual(
        data_read_write_service.get_write_call_args(), [[expected_request]]
    )
    server.stop()


if __name__ == "__main__":
  absltest.main()
