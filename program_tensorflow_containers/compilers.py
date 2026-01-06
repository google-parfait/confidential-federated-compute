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
import tensorflow as tf


# Turn on ahead-of-time graph optimization.
grappler_config_pb = tf.compat.v1.ConfigProto()
aggressive = grappler_config_pb.graph_options.rewrite_options.AGGRESSIVE
rewrite_options = grappler_config_pb.graph_options.rewrite_options
rewrite_options.memory_optimization = aggressive
rewrite_options.constant_folding = aggressive
rewrite_options.arithmetic_optimization = aggressive
rewrite_options.loop_optimization = aggressive
rewrite_options.function_optimization = aggressive


def compile_tf_to_call_dominant(
    comp: federated_language.framework.ConcreteComputation,
) -> federated_language.framework.ConcreteComputation:
  """Compile a TF-based computation to run on the program executor TEE."""
  comp_bb, _ = tff.tensorflow.replace_intrinsics_with_bodies(
      comp.to_building_block()
  )
  cdf_comp_bb = tff.framework.to_call_dominant(comp_bb)
  optimized_comp_bb, _ = tff.tensorflow.optimize_tensorflow_graphs(
    cdf_comp_bb, grappler_config_pb
  )
  # Currently this compiler step will cause errors in some federated programs
  # during ReduceDataset operations that complain about missing placeholders. So
  # we disable for now.
  # disabled_grappler_bb, _ = tff.tensorflow.transform_tf_call_ops_to_disable_grappler(
  #   optimized_comp_bb
  # )
  final_comp_bb, _ =  tff.tensorflow.transform_tf_add_ids(optimized_comp_bb)
  return federated_language.framework.ConcreteComputation.from_building_block(
      final_comp_bb
  )
