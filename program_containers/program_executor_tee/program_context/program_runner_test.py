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
from collections.abc import Callable
import unittest
from unittest import mock

from absl.testing import absltest
import federated_language
import numpy as np
import portpicker
from program_executor_tee.program_context import program_runner
from program_executor_tee.program_context.cc import fake_service_bindings_jax
import tensorflow_federated as tff
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2

_TRUSTED_PROGRAM_STRING = """
import tensorflow_federated as tff
import federated_language

def trusted_program(input_provider, external_service_handle):
    result1 = 1+2
    result2 = 1*2

    result1_val, _ = tff.framework.serialize_value(result1, federated_language.framework.infer_type(result1))
    result2_val, _ = tff.framework.serialize_value(result2, federated_language.framework.infer_type(result2))

    external_service_handle.release_unencrypted(result1_val.SerializeToString(), b"result1")
    external_service_handle.release_unencrypted(result2_val.SerializeToString(), b"result2")
    """


class ProgramRunnerTest(unittest.TestCase):

  # TODO: b/451714072 - Delete this test once the KMS migration is complete.
  def test_run_program_ledger(self):
    untrusted_root_port = portpicker.pick_unused_port()
    self.assertIsNotNone(untrusted_root_port, "Failed to pick an unused port.")
    data_read_write_service = (
        fake_service_bindings_jax.FakeDataReadWriteService()
    )
    server = fake_service_bindings_jax.FakeServer(
        untrusted_root_port, data_read_write_service, None
    )
    server.start()

    program_runner.run_program(
        initialize_fn=None,
        program=base64.b64encode(_TRUSTED_PROGRAM_STRING.encode("utf-8")),
        client_ids=[],
        client_data_directory="",
        model_id_to_zip_file={},
        outgoing_server_address=f"[::1]:{untrusted_root_port}",
        resolve_uri_to_tensor=lambda uri, key: tensor_pb2.TensorProto(),
        use_kms=False,
        release_unencrypted=lambda value, key: None,
    )

    released_data = data_read_write_service.get_released_data()
    self.assertEqual(len(released_data), 2)
    value_3, _ = tff.framework.serialize_value(
        3, federated_language.TensorType(np.int32)
    )
    self.assertEqual(
        released_data["result1"].encode("latin-1"), value_3.SerializeToString()
    )
    value_2, _ = tff.framework.serialize_value(
        2, federated_language.TensorType(np.int32)
    )
    self.assertEqual(
        released_data["result2"].encode("latin-1"), value_2.SerializeToString()
    )

    server.stop()

  def test_run_program(self):
    mock_release_fn = mock.MagicMock(spec=Callable)
    program_runner.run_program(
        initialize_fn=None,
        program=base64.b64encode(_TRUSTED_PROGRAM_STRING.encode("utf-8")),
        client_ids=[],
        client_data_directory="",
        model_id_to_zip_file={},
        outgoing_server_address="fake_address",
        resolve_uri_to_tensor=lambda uri, key: tensor_pb2.TensorProto(),
        use_kms=True,
        release_unencrypted=mock_release_fn,
    )

    value_3, _ = tff.framework.serialize_value(
        3, federated_language.TensorType(np.int32)
    )
    value_2, _ = tff.framework.serialize_value(
        2, federated_language.TensorType(np.int32)
    )
    self.assertEqual(mock_release_fn.call_count, 2)
    mock_release_fn.assert_has_calls([
        mock.call(value_3.SerializeToString(), b"result1"),
        mock.call(value_2.SerializeToString(), b"result2"),
    ])

  def test_run_program_without_trusted_program_function(self):
    program_string = """
def incorrectly_named_trusted_program(input_provider, external_service_handle):
    return 10
    """

    with self.assertRaises(ValueError) as context:
      program_runner.run_program(
          initialize_fn=None,
          program=base64.b64encode(program_string.encode("utf-8")),
          client_ids=[],
          client_data_directory="",
          model_id_to_zip_file={},
          outgoing_server_address="",
          resolve_uri_to_tensor=lambda uri, key: tensor_pb2.TensorProto(),
          use_kms=True,
          release_unencrypted=lambda value, key: None,
      )
    self.assertEqual(
        str(context.exception),
        "The provided program must have a "
        + program_runner.TRUSTED_PROGRAM_KEY
        + " function.",
    )


if __name__ == "__main__":
  absltest.main()
