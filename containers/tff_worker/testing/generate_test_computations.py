# Copyright 2024 Google LLC.
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
"""Binary for generating a serialized TFF computations for testing.

Run using:

  python3 -m venv venv && source venv/bin/activate
  pip install --upgrade pip
  pip install --upgrade tensorflow-federated==0.75.0
  python3 generate_test_computations.py
  deactivate
"""

import collections
import os

from absl import app
from absl import flags

from google.protobuf import text_format
import tensorflow as tf
import tensorflow_federated as tff

OUTPUT_DIR = flags.DEFINE_string('output_dir', '.', 'Output directory')

CLIENT_WORK_COMPUTATION = 'client_work_computation.txtpb'
AGGREGATION_COMPUTATION = 'aggregation_computation.txtpb'


@tff.tensorflow.computation(
    collections.OrderedDict(
        [('float', tff.TensorType(tf.float32.as_numpy_dtype, shape=[None]))]),
    tf.float32.as_numpy_dtype,
)
def tf_comp(
    example: collections.OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> collections.OrderedDict[str, tf.Tensor]:
  """Adds a broadcasted value to all floats in the example tensors."""
  result = collections.OrderedDict()
  for name, t in example.items():
    result[name] = tf.add(t, broadcasted_data)
  return result


# Define a federated computation using tf_comp to be performed per-client.
@tff.federated_computation(
    tff.FederatedType(
        collections.OrderedDict([
            ('float',
             tff.TensorType(tf.float32.as_numpy_dtype, shape=[None]))]),
        tff.CLIENTS,
    ),
    tff.FederatedType(tf.float32.as_numpy_dtype, tff.CLIENTS, all_equal=True),
)
def client_work_comp(
    example: collections.OrderedDict[str, tf.Tensor], broadcasted_data: float
) -> collections.OrderedDict[str, tf.Tensor]:
  """Performs `tf_comp` on each client."""
  return tff.federated_map(tf_comp, (example, broadcasted_data))


@tff.federated_computation(
    tff.FederatedType(tf.int32.as_numpy_dtype, tff.SERVER),
    tff.FederatedType(tf.int32.as_numpy_dtype, tff.CLIENTS)
)
def aggregation_comp(state: int, value: list[int]) -> int:
  """Scales then sums the outputs from applying client work at clients."""
  state_at_clients = tff.federated_broadcast(state)
  scaled_value = tff.federated_map(
      tff.tensorflow.computation(lambda x, y: x * y), (value, state_at_clients)
  )
  summed_value = tff.federated_sum(scaled_value)
  return summed_value


def generate_test_computations() -> None:
  """Generates serialized test computations and writes them out to files."""
  client_work_computation_text_proto = text_format.MessageToString(
      tff.framework.serialize_computation(client_work_comp))
  client_work_computation_filepath = os.path.join(
      OUTPUT_DIR.value, CLIENT_WORK_COMPUTATION)

  with open(client_work_computation_filepath, 'w') as f:
    f.write(client_work_computation_text_proto)

  aggregation_computation_text_proto = text_format.MessageToString(
      tff.framework.serialize_computation(aggregation_comp))
  aggregation_computation_filepath = os.path.join(
      OUTPUT_DIR.value, AGGREGATION_COMPUTATION)

  with open(aggregation_computation_filepath, 'w') as f:
    f.write(aggregation_computation_text_proto)


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  generate_test_computations()


if __name__ == '__main__':
  app.run(main)
