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
"""Binary for generating a serialized TFF computations for testing."""

import collections.abc
import os

from absl import app
from absl import flags

import tensorflow_federated as tff
from tff_worker.server.testing import test_computations

OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory',
                                 required=True)

CLIENT_WORK_COMPUTATION = 'client_work_computation.pb'
AGGREGATION_COMPUTATION = 'aggregation_computation.pb'


def generate_test_computations() -> None:
  """Generates serialized test computations and writes them out to files."""
  serialized_client_work_computation = tff.framework.serialize_computation(
      test_computations.client_work_comp).SerializeToString()
  client_work_computation_filepath = os.path.join(
      OUTPUT_DIR.value, CLIENT_WORK_COMPUTATION)

  with open(client_work_computation_filepath, 'wb') as f:
    f.write(serialized_client_work_computation)

  serialized_aggregation_computation = tff.framework.serialize_computation(
      test_computations.aggregation_comp).SerializeToString()
  aggregation_computation_filepath = os.path.join(
      OUTPUT_DIR.value, AGGREGATION_COMPUTATION)

  with open(aggregation_computation_filepath, 'wb') as f:
    f.write(serialized_aggregation_computation)


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  generate_test_computations()


if __name__ == '__main__':
  app.run(main)
