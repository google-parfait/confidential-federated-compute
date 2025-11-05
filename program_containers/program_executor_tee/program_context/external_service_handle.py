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

from fcp.confidentialcompute.python import external_service_handle
from fcp.confidentialcompute.python import external_service_handle
from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc
import grpc


class ExternalServiceHandle(external_service_handle.ExternalServiceHandle):
  """Helper class for releasing results to untrusted space."""

  def __init__(self, outgoing_server_address: str):
    """Establishes a channel to the DataReadWrite service."""
    super().__init__(outgoing_server_address)
    self._channel = grpc.insecure_channel(outgoing_server_address)
    self._stub = data_read_write_pb2_grpc.DataReadWriteStub(self._channel)

  def release_unencrypted(self, value: bytes, key: bytes) -> None:
    write_request = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id=key
            )
        ),
        commit=True,
        data=value,
    )
    self._stub.Write(iter([write_request]))

  def release_encrypted(
      self,
      value: bytes,
      key: bytes,
  ) -> None:
    """Releases an encrypted value to the external service."""
    raise NotImplementedError
