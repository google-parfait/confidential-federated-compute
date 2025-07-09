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

from fcp.protos.confidentialcompute import confidential_transform_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc
import federated_language
import grpc
import tensorflow_federated as tff


class ReleaseManager(
    federated_language.program.ReleaseManager[
        federated_language.program.ReleasableStructure, str
    ]
):
  """Helper class for releasing results to untrusted space."""

  def __init__(self, outgoing_server_address: str):
    """Establishes a channel to the DataReadWrite service."""
    self._channel = grpc.insecure_channel(outgoing_server_address)
    self._stub = data_read_write_pb2_grpc.DataReadWriteStub(self._channel)

  async def release(
      self, value: federated_language.program.ReleasableStructure, key: str
  ) -> None:
    serialized_value, _ = tff.framework.serialize_value(
        value, federated_language.framework.infer_type(value)
    )
    write_request = data_read_write_pb2.WriteRequest(
        first_request_metadata=confidential_transform_pb2.BlobMetadata(
            unencrypted=confidential_transform_pb2.BlobMetadata.Unencrypted(
                blob_id=key.encode()
            )
        ),
        commit=True,
        data=serialized_value.SerializeToString(),
    )
    self._stub.Write(iter([write_request]))
