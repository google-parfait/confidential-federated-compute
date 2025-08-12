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
from fcp.protos.confidentialcompute import file_info_pb2
import federated_language
from google.protobuf import any_pb2
from tensorflow_federated.proto.v0 import executor_pb2


def create_data_value(
    uri: str, key: str, type_spec: object, client_upload: bool = True
) -> executor_pb2.Value:
  """Creates a tff Value containing a file_info_pb2.FileInfo Data message."""
  content = any_pb2.Any()
  content.Pack(
      file_info_pb2.FileInfo(uri=uri, key=key, client_upload=client_upload)
  )
  return executor_pb2.Value(
      computation=federated_language.framework.Data(
          content=content,
          type_signature=type_spec,
      ).to_proto()
  )


def create_array_value(value: object, type_spec: object) -> executor_pb2.Value:
  """Creates a tff Value containing a Literal message."""
  return executor_pb2.Value(
      computation=federated_language.framework.Literal(
          value, type_spec
      ).to_proto()
  )


def parse_read_response_fn(
    read_response: data_read_write_pb2.ReadResponse, nonce: str, key: str
) -> executor_pb2.Value:
  """Parsing function to provide as an arg to TrustedContext for testing.

  This function assumes that any ReadResponses it is asked to parse contain
  unencrypted data that is a serialized tff Value proto. The nonce and key
  arguments are thus ignored in this implementation.
  """
  del nonce, key
  value = executor_pb2.Value()
  value.ParseFromString(read_response.data)
  return value
