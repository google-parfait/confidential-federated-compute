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

from fcp.protos.confidentialcompute import data_read_write_pb2
from fcp.protos.confidentialcompute import data_read_write_pb2_grpc


class FakeDataReadWriteServicer(data_read_write_pb2_grpc.DataReadWriteServicer):
  """A DataReadWriteServicer that records call args and returns empty responses.

  Using `mock_servicer = mock.create_autospec(DataReadWriteServicer)` and
  `mock_servicer.Write.return_value = WriteResponse()` appears to consume the
  input generator args such that they cannot be checked later via
  `mock_servicer.Write.assert_has_calls`. This class records the stream of
  WriteRequests during handling of the Write request so that they can be
  accessed later via `get_write_call_args`.
  """

  def __init__(self):
    self._write_call_args = []

  def Write(self, request_iterator, context):
    """Handles a stream of WriteRequests.

    Appends the stream of requests to _write_call_args and returns an empty
    WriteResponse.
    """
    del context
    self._write_call_args.append(list(request_iterator))
    return data_read_write_pb2.WriteResponse()

  def get_write_call_args(self) -> list[list[data_read_write_pb2.WriteRequest]]:
    return self._write_call_args
