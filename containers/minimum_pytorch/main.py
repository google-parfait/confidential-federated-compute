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

import contextlib
import random
import os
import time

from absl import app
from absl import flags
import numpy as np
import torch

from gemma import config
from gemma import gemma3_model
from gemma import gemma3_preprocessor

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', 'model.ckpt', 'Path to the checkpoint file.')
_VARIANT = flags.DEFINE_string('variant', '4b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
_OUTPUT_LEN = flags.DEFINE_integer(
    'output_len', 1024, 'Length of the output sequence.'
)
_SEED = flags.DEFINE_integer('seed', 12345, 'Random seed.')
_QUANT = flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')

# Define valid multimodal model variants
_VALID_MODEL_VARIANTS = ['4b', '12b', '27b_v3']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']


# Validator function for the 'variant' flag
def validate_variant(variant):
  if variant not in _VALID_MODEL_VARIANTS:
    raise ValueError(
        f'Invalid variant: {variant}. Valid variants are:'
        f' {_VALID_MODEL_VARIANTS}'
    )
  return True


# Validator function for the 'device' flag
def validate_device(device):
  if device not in _VALID_DEVICES:
    raise ValueError(
        f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}'
    )
  return True


# Register the validator for the 'variant' flag
flags.register_validator(
    'variant', validate_variant, message='Invalid model variant.'
)

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


def generate(model,
             device,
             prompts,
             output_len: int = 100,
             temperature: float|None = 1.0,
             top_p: float = 0.95,
             top_k: int = 64):
    """Generates responses for given prompts using Gemma model."""
    # Inference only.
    processing_result = gemma3_preprocessor.tokenize_raw_input(
            model.tokenizer, prompts, model.config, output_len, device
        )
    batch_size = processing_result["batch_size"]
    user_input_token_ids = processing_result["user_input_token_ids"]
    image_batch = processing_result["image_batch"]
    min_prompt_len = processing_result["min_prompt_len"]
    max_prompt_len = processing_result["max_prompt_len"]
    total_seq_len = processing_result["max_seq_len"]
    image_presence_mask = processing_result["image_presence_mask"]

    # Create attention mask.
    min_dtype = torch.finfo(model.dtype).min
    if model.config.sliding_window_size is None:
      raise ValueError('gemma 3 model requires sliding_window size')
    boolean_mask, local_boolean_mask = model.create_attention_mask(user_input_token_ids, total_seq_len)
    mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
    local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()

    kv_caches = []
    for _ in range(model.config.num_hidden_layers):
      size = (batch_size, total_seq_len, model.config.num_key_value_heads,
                    model.config.head_dim)
      dtype = model.config.get_dtype()
      k_cache = torch.zeros(size=size, dtype=dtype, device=device)
      v_cache = torch.zeros(size=size, dtype=dtype, device=device)
      kv_caches.append((k_cache, v_cache))

    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            model.tokenizer.pad_id,
                                            dtype=torch.int64, device=device)
    token_ids_tensor = user_input_token_ids.to(device)
    for i in range(batch_size):
      p = user_input_token_ids[i]
      input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

    input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
    prompt_mask_tensor = token_ids_tensor != model.tokenizer.pad_id
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.

    start_time = time.monotonic()
    for i in range(total_seq_len - min_prompt_len):
      next_token_ids, _ = model(
                input_token_ids=input_token_ids_tensor,
                image_patches=image_batch,
                image_presence_mask=image_presence_mask,
                input_positions=input_positions_tensor,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )
      curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
      curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
      output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
      token_ids_tensor.index_copy_(1, output_index, output_token_ids)

      input_token_ids_tensor = output_token_ids
      input_positions_tensor = output_index.unsqueeze(dim=-1)
      curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
      curr_local_mask_tensor = local_mask_tensor.index_select(
                2, input_positions_tensor
            ) if local_mask_tensor is not None else None
      output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
      output_index = output_index + 1
      image_batch = None
      image_presence_mask = None
      # This condition only works with single prompt
      if output_token_ids.item() == model.tokenizer.eos_id :
        break

    end_time = time.monotonic();

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    tps = len(token_ids[0]) / (end_time - start_time);
    print(f'Number of tokens: {len(token_ids)}; elapsed seconds: {end_time - start_time}')
    print(f'Performance: {tps} t/s', )

    results = []
    for i, tokens in enumerate(token_ids):
      output = tokens
      if model.tokenizer.eos_id in output:
        eos_index = output.index(model.tokenizer.eos_id)
        output = output[:eos_index]
      results.append(model.tokenizer.decode(output))

    return results


def main(_):
  # Construct the model config.
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float32'
  model_config.quant = _QUANT.value
  model_config.tokenizer = 'tokenizer.model'

  # Seed random.
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)

  # Create the model and load the weights.
  device = torch.device(_DEVICE.value)
  with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    model.load_state_dict(torch.load(_CKPT.value)['model_state_dict'])
    # model.load_weights(_CKPT.value)
    model = model.to(device).eval()
  print('Model loading done')

  # Generate text only.
  result = generate(
      model,
      device,
      [['Tell three facts about cats.']],
      output_len=_OUTPUT_LEN.value,
  )

  # Print the results.
  print('======================================')
  print(f'Text only RESULT: {result}')
  print('======================================')


if __name__ == '__main__':
  app.run(main)