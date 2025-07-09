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
"""Binary for generating serialized TFF computations for testing.

Run using:

  python3 -m venv venv && source venv/bin/activate
  pip install --upgrade pip
  pip install --upgrade tensorflow-federated==0.75.0
  python3 generate_test_computations.py
  deactivate
"""

import collections
import numpy as np
import os

from absl import app
from absl import flags

from google.protobuf import text_format
import tensorflow_federated as tff

OUTPUT_DIR = flags.DEFINE_string('output_dir', '.', 'Output directory')

NO_ARGUMENT_FUNCTION = 'no_argument_function.txtpb'
SERVER_DATA_FUNCTION = 'server_data_function.txtpb'
CLIENT_DATA_FUNCTION = 'client_data_function.txtpb'

@tff.federated_computation
def no_argument_comp():
  return tff.federated_value(10, tff.SERVER)

@tff.tf_computation(np.int32, np.int32)
def add(x, y):
  return x + y

@tff.tf_computation(np.int32)
def identity(x):
  return x

@tff.tf_computation(np.int32)
def scale(x):
  return x * 10

@tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
def client_data_comp(client_data):
  return tff.federated_aggregate(client_data, 0, add, add, identity)

@tff.federated_computation(tff.FederatedType(np.int32, tff.SERVER))
def server_data_comp(server_state):
  scaled_server_state = tff.federated_map(scale, server_state)
  broadcasted_server_state = tff.federated_broadcast(scaled_server_state)
  summed_broadcast = tff.federated_aggregate(
      broadcasted_server_state, 0, add, add, identity
  )
  return scaled_server_state, summed_broadcast

def generate_test_computations() -> None:
  """Generates serialized test computations and writes them out to files."""
  no_argument_function_text_proto = text_format.MessageToString(
      tff.framework.serialize_computation(no_argument_comp))
  no_argument_function_filepath = os.path.join(
      OUTPUT_DIR.value, NO_ARGUMENT_FUNCTION)

  with open(no_argument_function_filepath, 'w') as f:
    f.write(no_argument_function_text_proto)

  server_data_function_text_proto = text_format.MessageToString(
      tff.framework.serialize_computation(server_data_comp))
  server_data_function_filepath = os.path.join(
      OUTPUT_DIR.value, SERVER_DATA_FUNCTION)

  with open(server_data_function_filepath, 'w') as f:
    f.write(server_data_function_text_proto)

  client_data_function_text_proto = text_format.MessageToString(
      tff.framework.serialize_computation(client_data_comp))
  client_data_function_filepath = os.path.join(
      OUTPUT_DIR.value, CLIENT_DATA_FUNCTION)

  with open(client_data_function_filepath, 'w') as f:
    f.write(client_data_function_text_proto)


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  generate_test_computations()


if __name__ == '__main__':
  app.run(main)
