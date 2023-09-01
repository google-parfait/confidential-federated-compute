# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for creating Tensorflow checkpoints for testing."""
import uuid
from typing import List
import tensorflow as tf
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2 as worker_pb2
from tensorflow.core.framework import types_pb2


def create_checkpoint_bytes(
    names: List[str], tensors: List[tf.Tensor]
) -> bytes:
  tmp_path = f'ram://{uuid.uuid4()}.ckpt'
  try:
    tf.raw_ops.SaveSlices(
        filename=tmp_path,
        tensor_names=names,
        shapes_and_slices=[''] * len(tensors),
        data=tensors,
    )
    # Read the checkpoint bytes from the temporary file.
    with tf.io.gfile.GFile(tmp_path, 'rb') as f:
      return f.read()
  finally:
    tf.io.gfile.remove(tmp_path)

def column_config(
    name: str, data_type: types_pb2.DataType
) -> worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint.FedSqlColumn:
  return (
      worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint.FedSqlColumn(
          name=name, data_type=data_type
      )
  )
