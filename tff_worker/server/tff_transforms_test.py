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

import unittest
import tensorflow as tf
import tensorflow_federated as tff
from tff_worker.server import tff_transforms


class TffTransformsTest(unittest.TestCase):

  def test_aggregate(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    serialized_expected_output, _ = tff.framework.serialize_value(
        tf.constant('The quick brown fox'), tf.string
    )
    self.assertEqual(
        serialized_expected_output.SerializeToString(),
        tff_transforms.aggregate(
            serialized_aggregation_comp.SerializeToString(),
            serialized_temp_state.SerializeToString(),
            [
                serialized_input_a.SerializeToString(),
                serialized_input_b.SerializeToString(),
                serialized_input_c.SerializeToString(),
            ],
        ),
    )

  def test_aggregate_unparseable_computation(self):
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          b'invalid',
          serialized_temp_state.SerializeToString(),
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn('Could not parse serialized computation', str(e.exception))

  def test_aggregate_unparseable_state(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          serialized_aggregation_comp.SerializeToString(),
          b'invalid_temp_state',
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Could not parse serialized temporary state', str(e.exception)
    )

  def test_aggregate_unparseable_client_input(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          serialized_aggregation_comp.SerializeToString(),
          serialized_temp_state.SerializeToString(),
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              b'invalid_client_input',
          ],
      )
    self.assertIn('Could not parse serialized client input', str(e.exception))

  def test_aggregate_computation_wrong_num_args(self):
    aggregation_comp = tff.tf_computation(
        lambda x: tf.strings.join([x, 'fox'], separator=' ', name=None),
        (tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          serialized_aggregation_comp.SerializeToString(),
          serialized_temp_state.SerializeToString(),
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Expected aggregation computation input type', str(e.exception)
    )

  def test_aggregate_computation_wrong_temp_state_type(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant(1.0), tf.float32
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant('fox'), tf.string
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          serialized_aggregation_comp.SerializeToString(),
          serialized_temp_state.SerializeToString(),
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Type `float32` is not assignable to type `string`', str(e.exception)
    )

  def test_aggregate_computation_wrong_client_input_type(self):
    aggregation_comp = tff.tf_computation(
        lambda x, y: tf.strings.join([x, y], separator=' ', name=None),
        (tf.string, tf.string),
    )
    serialized_aggregation_comp = tff.framework.serialize_computation(
        aggregation_comp
    )
    serialized_temp_state, _ = tff.framework.serialize_value(
        tf.constant('The'), tf.string
    )
    serialized_input_a, _ = tff.framework.serialize_value(
        tf.constant('quick'), tf.string
    )
    serialized_input_b, _ = tff.framework.serialize_value(
        tf.constant('brown'), tf.string
    )
    serialized_input_c, _ = tff.framework.serialize_value(
        tf.constant(1.0), tf.float32
    )
    with self.assertRaises(TypeError) as e:
      tff_transforms.aggregate(
          serialized_aggregation_comp.SerializeToString(),
          serialized_temp_state.SerializeToString(),
          [
              serialized_input_a.SerializeToString(),
              serialized_input_b.SerializeToString(),
              serialized_input_c.SerializeToString(),
          ],
      )
    self.assertIn(
        'Type `float32` is not assignable to type `string`', str(e.exception)
    )


if __name__ == '__main__':
  unittest.main(verbosity=2)
