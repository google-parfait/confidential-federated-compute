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

import federated_language
import tensorflow_federated as tff


def compile_tf_to_call_dominant(
    comp: federated_language.framework.ConcreteComputation,
) -> federated_language.framework.ConcreteComputation:
  """Compile a TF-based computation to run on the program executor TEE."""
  comp_bb, _ = tff.tensorflow.replace_intrinsics_with_bodies(
      comp.to_building_block()
  )
  comp_bb = tff.framework.to_call_dominant(comp_bb)
  return federated_language.framework.ConcreteComputation.from_building_block(
      comp_bb
  )
