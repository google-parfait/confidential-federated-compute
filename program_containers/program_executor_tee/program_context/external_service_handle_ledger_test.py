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
from program_executor_tee.program_context import external_service_handle_ledger
from program_executor_tee.program_context.cc import fake_service_bindings_jax
import tensorflow_federated as tff


class ExternalServiceHandleForLedgerTest(unittest.TestCase):

  def test_unencrypted_release(self):
    untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(untrusted_root_port, "Failed to pick an unused port.")
    data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    server = fake_service_bindings_jax.FakeServer(
        untrusted_root_port, data_read_write_service, None
    )
    server.start()

    handle = external_service_handle_ledger.ExternalServiceHandleForLedger(
        f"[::1]:{untrusted_root_port}"
    )
    result_uri = b"my_result"
    value, _ = tff.framework.serialize_value(
        5, federated_language.TensorType(np.int32)
    )
    serialized_value = value.SerializeToString()
    handle.release_unencrypted(serialized_value, result_uri)

    released_data = data_read_write_service.get_released_data()
    self.assertEqual(len(released_data), 1)
    self.assertEqual(
        released_data["my_result"].encode("latin-1"), serialized_value
    )

    server.stop()


if __name__ == "__main__":
  absltest.main()
