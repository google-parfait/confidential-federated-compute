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

from fcp.protos.confidentialcompute import file_info_pb2
from tensorflow_federated.proto.v0 import executor_pb2


def replace_datas(
    value: executor_pb2.Value,
    resolve_fileinfo_to_tff_value_fn: Callable[
        [file_info_pb2.FileInfo], executor_pb2.Value
    ],
) -> executor_pb2.Value:
  """Replaces any Data pointers in the input with resolved values.

  Iterates through the input and upon finding any Data pointers, calls
  the provided resolution function and replaces the Data pointer with
  the resolved value. Any Data pointers are expected to include a content field
  that holds a file_info_pb2.FileInfo message.

  Args:
    value: A executor_pb2.Value that may or may not contain Data pointers that
      hold a file_info_pb2.FileInfo message.
    resolve_fileinfo_to_tff_value_fn: A function that can be used to resolve a
      file_info_pb2.FileInfo message into a executor_pb2.Value message.

  Returns:
    A executor_pb2.Value with no Data pointers.

  Raises:
    ValueError: If a Data pointer cannot be parsed or resolved.
  """
  if value.HasField("computation"):
    if value.computation.HasField("data"):
      file_info = file_info_pb2.FileInfo()
      if not value.computation.data.content.Unpack(file_info):
        raise ValueError(
            "Unable to unpack Data pointer content to file_info_pb2.FileInfo"
        )
      try:
        return resolve_fileinfo_to_tff_value_fn(file_info)
      except Exception as e:
        raise ValueError(f"Unable to resolve Data pointer: {e}")
    return value
  if value.HasField("struct"):
    elements = []
    for element in value.struct.element:
      elements.append(
          executor_pb2.Value.Struct.Element(
              name=element.name,
              value=replace_datas(
                  element.value, resolve_fileinfo_to_tff_value_fn
              ),
          )
      )
    return executor_pb2.Value(
        struct=executor_pb2.Value.Struct(element=elements)
    )
  if value.HasField("federated"):
    elements = []
    for element in value.federated.value:
      elements.append(replace_datas(element, resolve_fileinfo_to_tff_value_fn))
    return executor_pb2.Value(
        federated=executor_pb2.Value.Federated(
            type=value.federated.type, value=elements
        )
    )
  return value
