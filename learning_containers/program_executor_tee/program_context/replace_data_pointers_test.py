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

from fcp.protos.confidentialcompute import file_info_pb2
import federated_language
from google.protobuf import any_pb2
from learning_containers.program_executor_tee.program_context import replace_data_pointers
from learning_containers.program_executor_tee.program_context import test_helpers
import numpy as np
from tensorflow_federated.proto.v0 import executor_pb2


class ReplaceDataPointersTest(unittest.TestCase):

  def setUp(self):
    client_tensor_type = federated_language.TensorType(np.str_)

    self.client_1_data = test_helpers.create_data_value(
        "client_1", "mykey", client_tensor_type
    )
    self.client_2_data = test_helpers.create_data_value(
        "client_2", "mykey", client_tensor_type
    )
    self.client_3_data = test_helpers.create_data_value(
        "client_3", "mykey", client_tensor_type
    )

    self.client_1_array = test_helpers.create_array_value(
        "client_1", client_tensor_type
    )
    self.client_2_array = test_helpers.create_array_value(
        "client_2", client_tensor_type
    )
    self.client_3_array = test_helpers.create_array_value(
        "client_3", client_tensor_type
    )

    self.already_resolved_array = test_helpers.create_array_value(
        "already_resolved", client_tensor_type
    )

    self.uri_to_value_map = {
        "client_1": self.client_1_array,
        "client_2": self.client_2_array,
        "client_3": self.client_3_array,
    }

  def fileinfo_lookup(self, fileinfo: file_info_pb2.FileInfo):
    if fileinfo.uri not in self.uri_to_value_map:
      raise ValueError("uri not in map")
    return self.uri_to_value_map[fileinfo.uri]

  def test_replace_datas_federated(self):
    # Create a federated tff Value that contains a mix of values that are data pointers and values that are not.
    arg = executor_pb2.Value()
    arg.federated.type.placement.value.uri = federated_language.CLIENTS.uri
    arg.federated.type.all_equal = False
    arg.federated.value.append(self.client_1_data)
    arg.federated.value.append(self.already_resolved_array)
    arg.federated.value.append(self.client_3_data)

    resolved_arg = replace_data_pointers.replace_datas(
        arg, self.fileinfo_lookup
    )

    expected_resolved_arg = executor_pb2.Value()
    expected_resolved_arg.federated.type.placement.value.uri = (
        federated_language.CLIENTS.uri
    )
    expected_resolved_arg.federated.type.all_equal = False
    expected_resolved_arg.federated.value.append(self.client_1_array)
    expected_resolved_arg.federated.value.append(self.already_resolved_array)
    expected_resolved_arg.federated.value.append(self.client_3_array)

    self.assertEqual(resolved_arg, expected_resolved_arg)

  def test_replace_datas_struct(self):
    arg = executor_pb2.Value()
    arg.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element1", value=self.client_1_data
        )
    )
    inner_struct = executor_pb2.Value()
    inner_struct.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element2", value=self.already_resolved_array
        )
    )
    inner_struct.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element3", value=self.client_3_data
        )
    )
    arg.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="inner_struct", value=inner_struct
        )
    )

    resolved_arg = replace_data_pointers.replace_datas(
        arg, self.fileinfo_lookup
    )

    expected_resolved_arg = executor_pb2.Value()
    expected_resolved_arg.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element1", value=self.client_1_array
        )
    )
    expected_resolved_inner_struct = executor_pb2.Value()
    expected_resolved_inner_struct.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element2", value=self.already_resolved_array
        )
    )
    expected_resolved_inner_struct.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="element3", value=self.client_3_array
        )
    )
    expected_resolved_arg.struct.element.append(
        executor_pb2.Value.Struct.Element(
            name="inner_struct", value=expected_resolved_inner_struct
        )
    )

    self.assertEqual(resolved_arg, expected_resolved_arg)

  def test_replace_datas_uri_resolution_error(self):
    arg = test_helpers.create_data_value(
        "client_unregistered", "mykey", federated_language.TensorType(np.str_)
    )

    with self.assertRaises(ValueError) as context:
      replace_data_pointers.replace_datas(arg, self.fileinfo_lookup)
    self.assertIn(
        "Unable to resolve Data pointer: uri not in map",
        str(context.exception),
    )

  def test_replace_datas_missing_fileinfo(self):
    arg = executor_pb2.Value(
        computation=federated_language.framework.Data(
            content=any_pb2.Any(),
            type_signature=federated_language.TensorType(np.str_),
        ).to_proto()
    )

    with self.assertRaises(ValueError) as context:
      replace_data_pointers.replace_datas(arg, self.fileinfo_lookup)
    self.assertIn(
        "Unable to unpack Data pointer content to file_info_pb2.FileInfo",
        str(context.exception),
    )


if __name__ == "__main__":
  unittest.main()
