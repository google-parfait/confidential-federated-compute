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
import unittest
from unittest import mock

from absl.testing import absltest
from program_executor_tee.program_context import external_service_handle


class ExternalServiceHandleTest(unittest.TestCase):

  def test_unencrypted_release(self):
    mock_release_fn = mock.MagicMock(spec=Callable)
    handle = external_service_handle.ExternalServiceHandle(
        "fake_address", mock_release_fn
    )

    test_value = b"my_data"
    test_key = b"my_key"
    handle.release_unencrypted(test_value, test_key)

    mock_release_fn.assert_called_once()
    mock_release_fn.assert_called_with(test_value, test_key)


if __name__ == "__main__":
  absltest.main()
