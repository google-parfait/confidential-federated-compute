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
from typing import List

import jax.numpy as jnp
import sentencepiece


MODEL_PATH = 'model.ckpt'


# Decode every input string into tokens, and count the most frequent token.
# This method intentionally uses sentencepiece and jax Python API for demo purposes,
# not for performance or functional reasons.
def find_most_frequent_token(inputs: List[str]):
    sp_model = sentencepiece.SentencePieceProcessor()
    sp_model.Load(MODEL_PATH)

    tokenized_inputs = [sp_model.EncodeAsIds(s) for s in inputs]
    flat_array = jnp.concatenate(jnp.array(tokenized_inputs))
    unique_vals, counts = jnp.unique(flat_array, return_counts=True)
    max_count_index = jnp.argmax(counts)
    most_frequent_val = unique_vals[max_count_index]
    return sp_model.DecodeIds(most_frequent_val.tolist())
