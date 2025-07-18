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

from collections.abc import Sequence
import os
import shutil

from absl import app
import tensorflow as tf
import tensorflow_federated as tff


def build_linear_regression_keras_functional_model(feature_dims=2):
  """Build a linear regression `tf.keras.Model` using the functional API."""
  a = tf.keras.layers.Input(shape=(feature_dims,), dtype=tf.float32)
  b = tf.keras.layers.Dense(
      units=1,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      activation=None,
  )(a)
  return tf.keras.Model(inputs=a, outputs=b)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  keras_model = build_linear_regression_keras_functional_model()
  dataset = tf.data.Dataset.from_tensor_slices((
      [[1.0, 2.0], [3.0, 4.0]],
      [[5.0], [6.0]],
  )).batch(2)
  functional_model = tff.learning.models.functional_model_from_keras(
      keras_model,
      loss_fn=tf.keras.losses.MeanSquaredError(),
      input_spec=dataset.element_spec,
  )
  path = '/tmp/functional_model/'
  saved_model_path = os.path.join(path, 'test_model')
  tff.learning.models.save_functional_model(
      functional_model, saved_model_path
  )
  zip_file_path = os.path.join(path, 'model1')
  shutil.make_archive(zip_file_path, 'zip', saved_model_path)


if __name__ == '__main__':
  app.run(main)
