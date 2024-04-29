# Copyright 2024 Google LLC
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

'''CLI Utility for Running Inference on a Single Stream'''

import jax

import maxengine
from jetstream.engine import token_utils
from transformers import AutoTokenizer
from jax import numpy as jnp

import os
import pyconfig
import sys
from jax_smi import initialise_tracking
import json
initialise_tracking()

def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()
  print("LOADED PARAMS")
  with open("/home/ljy/out.txt", "a+") as f:
    f.write("PARAMS\n")
    f.write(str(params))
  # print("PARAMS:")
  # print(params)

  text = config.prompt
  # metadata = engine.get_tokenizer()
  
  # vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  # tokenizer = vocab.tokenizer
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'right'
  tokenized = tokenizer(text, padding = "max_length", max_length = config.max_prefill_predict_length, return_tensors = "jax")
  # tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True,
  #                                                    prefill_lengths=[config.max_prefill_predict_length])
  tokens = tokenized.input_ids[1]
  true_length = jnp.count_nonzero(tokenized.attention_mask[1])
  print(tokens)
  print(true_length)
  assert tokens.size <= config.max_prefill_predict_length, "can't take too many tokens"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  slot=0

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  print(results)
  output = tokenizer.decode(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert output==config.autoregressive_decode_assert, \
    f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"

def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  # print(cfg.real_vocab_size)
  main(cfg)
