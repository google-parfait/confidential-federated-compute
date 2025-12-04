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

from sentence_transformers import SentenceTransformer

# Bind the model lifecycle to the interpreter so we don't manage this on the C++ side.
model = None

def initialize_model(artifacts_path: str) -> bool:
  """Initialize the sentence transformer model

  Args:
    artifacts_path: Absolute path to the model artifact directory. It's expected to have multiple
    files inside the directory including the configs and weights for both the model and the tokenizer.

  Returns:
    Whether the model initialization succeeded.

  """
  global model
  model =  SentenceTransformer(artifacts_path)
  return model is not None


def encode(input:list[str], prompt:str|None = None) -> list[list[float]]:
  """Generate embeddings for a batch of samples

  Args:
    input: a list of strings to encode.
    prompt: an optional string used to guide the embedding generation.

  Returns:
    A list of embeddings for the input strings. Each embedding is a list of float.
  """
  global model
  return model.encode(input, prompt=prompt)
