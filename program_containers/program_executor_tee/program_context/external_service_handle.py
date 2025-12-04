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
from fcp.confidentialcompute.python import external_service_handle


class ExternalServiceHandle(external_service_handle.ExternalServiceHandle):
  """Helper class for releasing results to untrusted space."""

  def __init__(
      self,
      outgoing_server_address: str,
      release_unencrypted_fn: Callable[[bytes, bytes], None],
  ):
    """Establishes a channel to the DataReadWrite service."""
    super().__init__(outgoing_server_address)
    self.release_unencrypted_fn = release_unencrypted_fn

  def release_unencrypted(self, value: bytes, key: bytes) -> None:
    """Releases an unencrypted value to the external service."""
    self.release_unencrypted_fn(value, key)

  def release_encrypted(
      self,
      value: bytes,
      key: bytes,
  ) -> None:
    """Releases an encrypted value to the external service."""
    raise NotImplementedError
