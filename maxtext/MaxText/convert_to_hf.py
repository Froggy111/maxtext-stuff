
"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

r"""Convert weights from a MaxText model to a HuggingFace model.

Usage:

Get MaxText model weights from a MaxText run

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

python3 MaxText/llama_or_mistral_orbax_to_huggingface.py MaxText/configs/base.yml 
            base_output_directory=path/to/saving/intermediate_MaxText_files
            load_parameters_path=/path/to/MaxText/checkpoint run_name=<your run name> model_name=<llama2 or mistral> 
            hf_model_path=/local/path/to/save/HF/model/to

Note that we are saving the converted HuggingFace model to a local path. You can write to a GCS location by mounting
the GCS bucket as a local path using `setup_gcsfuse.sh`, but remember to mount as read+write.
"""

import jax
# jax.distributed.initialize()
# max_logging.log("initialised jax distributed")
from typing import Sequence
import torch
from tqdm import tqdm
from absl import app
import numpy as np
import pyconfig
import max_utils
import max_logging
from jax.sharding import Mesh
import max_logging
import checkpointing
from generate_param_only_checkpoint import _read_train_checkpoint
import llama_or_mistral_ckpt
from transformers import LlamaForCausalLM, MistralForCausalLM, AutoConfig, AutoModelForCausalLM, MixtralForCausalLM
from jax_smi import initialise_tracking
import sys
import os
from accelerate import init_empty_weights

def unpermute_from_match_maxtext_rope(arr):
  """
  Function to get the RoPE values in correct ordering
  """
  split_size = arr.shape[-1] // 2  # Assuming half for evens, half for odds
  evens = arr[..., :split_size]
  odds = arr[..., split_size:]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim-1)

def reverse_scale(arr,scale):
  """
  MaxText has the scaling factor included into the weights, 
  we reverse it when writing out the HuggingFace checkpoint
  """
  return arr * np.sqrt(scale)

def load_hf_model(model_size):
  """
  Load the model that we are interested in from HuggingFace

  """
  with init_empty_weights():
    max_logging.log("loading model")
    if model_size == "llama2-7b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    elif model_size == "mistral-7b":
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    elif model_size == "mixtral-8x22b":
        # config = AutoConfig.from_pretrained("/home/ljy/shared/Mixtral-8x22B-v0.1-resized-embeddings")
        model = MixtralForCausalLM.from_pretrained("/home/ljy/shared/Mixtral-8x22B-v0.1-resized-embeddings", torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
    else:
        raise NotImplementedError
    max_logging.log("loaded model")
    max_logging.log(model)
    return model

def load_model_state(config):
  """
  Loads the MaxText model's TrainState from the Orbax checkpoint
  """
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Create a checkpoint manager to load decode checkpoint at config.checkpoint_dir
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.load_parameters_path,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
  )

  # Read training state from config.load_paramaters_path
  max_logging.log(f"Read training checkpoint from: {config.load_parameters_path}")
#   training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
  params = _read_train_checkpoint(config, checkpoint_manager, mesh)
#   return training_state
  return params


def convert_state_to_hf(params, model_size):
  """
  Port the parameters from the Orbax training_state into the hf_model
  """

  if model_size not in llama_or_mistral_ckpt.MODEL_PARAMS_DICT:
    raise NotImplementedError
  # Load the model specific parameters
  model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params['num_layers']
  base_num_query_heads = model_params['num_heads']
  head_dim = model_params['dims_per_head']
  base_num_kv_heads = model_params['num_kv_heads']
  num_experts = model_params['num_experts'] if 'num_experts' in model_params else None

#   max_logging.log(params['params']['decoder']['layers']['gate'])


  hf_model_params = {}
  max_logging.log(params['params']["decoder"]["logits_dense"]["kernel"].shape)
  emb_idx = params['params']["decoder"]["logits_dense"]["kernel"].shape.index(32008)
  if emb_idx == 0:
    params['params']["decoder"]["logits_dense"]["kernel"] = params['params']["decoder"]["logits_dense"]["kernel"][:32002, ...]
  if emb_idx == 1:
    params['params']["decoder"]["logits_dense"]["kernel"] = params['params']["decoder"]["logits_dense"]["kernel"][:, :32002, ...]
  if emb_idx == 2:
    params['params']["decoder"]["logits_dense"]["kernel"] = params['params']["decoder"]["logits_dense"]["kernel"][:, :, :32002, ...]
  if emb_idx == 3:
    params['params']["decoder"]["logits_dense"]["kernel"] = params['params']["decoder"]["logits_dense"]["kernel"][:, :, :, :32002, ...]
  print(params['params']["decoder"]["logits_dense"]["kernel"].shape)
  hf_model_params["lm_head.weight"] = torch.tensor(np.array(
      params['params']["decoder"]["logits_dense"]["kernel"].T, dtype=np.float32),
      dtype=torch.bfloat16
  )
  hf_model_params["model.norm.weight"] = torch.tensor(np.array(
      params['params']["decoder"]["decoder_norm"]["scale"].reshape(base_num_query_heads * head_dim), dtype=np.float32),
      dtype=torch.bfloat16
  )

  #Port the embedding weights
  max_logging.log(params['params']['token_embedder']['embedding'].shape)
  emb_idx = params['params']['token_embedder']['embedding'].shape.index(32128)
  if emb_idx == 0:
    params['params']['token_embedder']['embedding'] = params['params']['token_embedder']['embedding'][:32002, ...]
  if emb_idx == 1:
    params['params']['token_embedder']['embedding'] = params['params']['token_embedder']['embedding'][:, :32002, ...]
  if emb_idx == 2:
    params['paras']['token_embedder']['embedding'] = params['params']['token_embedder']['embedding'][:, :, :32002, ...]
  if emb_idx == 3:
    params['params']['token_embedder']['embedding'] = params['params']['token_embedder']['embedding'][:, :, :, :32002, ...]
  hf_model_params["model.embed_tokens.weight"] = torch.tensor(
                    np.array(params['params']['token_embedder']['embedding'], dtype=np.float32),
                                    dtype=torch.bfloat16)
  del params['params']['token_embedder']['embedding']

  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting self attention"):
    #Attention layers
    hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"] = torch.tensor(np.array(
        unpermute_from_match_maxtext_rope(
          reverse_scale(
            params['params']["decoder"]["layers"]["self_attention"]["query"]["kernel"][:, layer_int, :, :]
            ,head_dim
            )
            ).reshape(base_num_query_heads * head_dim,base_num_query_heads * head_dim).T, dtype=np.float32),
        dtype=torch.bfloat16
    )
  
    hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"] = torch.tensor(np.array(
        unpermute_from_match_maxtext_rope(
          params['params']["decoder"]["layers"]["self_attention"]["key"]["kernel"][:, layer_int, :, :]
          ).reshape(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T, dtype=np.float32),
        dtype=torch.bfloat16
    )
    hf_model_params[f"model.layers.{layer_int}.self_attn.v_proj.weight"] = torch.tensor(np.array(
         params['params']["decoder"]["layers"]["self_attention"]["value"]["kernel"][:, layer_int, :, :]
         .reshape(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T, dtype=np.float32),
         dtype=torch.bfloat16
    )
    hf_model_params[f"model.layers.{layer_int}.self_attn.o_proj.weight"] = torch.tensor(np.array(
        params['params']["decoder"]["layers"]["self_attention"]["out"]["kernel"][:, layer_int, :, :]
        .reshape(base_num_query_heads * head_dim,base_num_query_heads * head_dim).T, dtype=np.float32),
        dtype=torch.bfloat16
    )
  del params['params']["decoder"]["layers"]["self_attention"]["query"]["kernel"]
  del params['params']["decoder"]["layers"]["self_attention"]["value"]["kernel"]
  del params['params']["decoder"]["layers"]["self_attention"]["out"]["kernel"]

  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting MoE gate"):
    #MLP Layers
    hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.gate.weight"] = torch.tensor(np.array(
        params['params']['decoder']['layers']['gate']['kernel'][:,layer_int,:].T, dtype=np.float32),
        dtype=torch.bfloat16
    )
  del params['params']['decoder']['layers']['gate']
  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting MLP w1"):
    for k in range(num_experts):
      hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w1.weight"] = torch.tensor(np.array(
          params['params']["decoder"]["layers"][f"mlp_{k}"]["wi_0"]["kernel"][:,layer_int,:].T, dtype=np.float32),
          dtype=torch.bfloat16
      )
  for k in range(num_experts):
    del params['params']["decoder"]["layers"][f"mlp_{k}"]["wi_0"]["kernel"]
  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting MLP w3"):
    for k in range(num_experts):
      hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w3.weight"] = torch.tensor(np.array(
          params['params']["decoder"]["layers"][f"mlp_{k}"]["wi_1"]["kernel"][:,layer_int,:].T, dtype=np.float32),
          dtype=torch.bfloat16
      )
  for k in range(num_experts):
    del params['params']["decoder"]["layers"][f"mlp_{k}"]["wi_1"]["kernel"]
  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting MLP w2"):
    for k in range(num_experts):
      hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w2.weight"] = torch.tensor(np.array(
          params['params']["decoder"]["layers"][f"mlp_{k}"]["wo"]["kernel"][:,layer_int,:].T, dtype=np.float32),
          dtype=torch.bfloat16
      )
  for k in range(num_experts):
    del params['params']["decoder"]["layers"][f"mlp_{k}"]["wo"]["kernel"]
  

  for layer_int in tqdm(range(base_num_decoder_layers),desc="Porting layernorms and LM head"):
    #Pre/post attention layer norm
    hf_model_params[f"model.layers.{layer_int}.input_layernorm.weight"] = torch.tensor(np.array(
      params['params']["decoder"]["layers"]["pre_self_attention_layer_norm"]["scale"][:,layer_int]
      .reshape(base_num_query_heads * head_dim), dtype=np.float32),
      dtype=torch.bfloat16
    )
    hf_model_params[f"model.layers.{layer_int}.post_attention_layernorm.weight"] = torch.tensor(np.array(
      params['params']["decoder"]["layers"]["post_self_attention_layer_norm"]["scale"][:,layer_int]
      .reshape(base_num_query_heads * head_dim), dtype=np.float32),
      dtype=torch.bfloat16
    )
  #LM head and layernorm

  return hf_model_params



def convert_orbax_hf(hf_model_path, config):
  """
  Landing function to convert MaxText model's checkpoint to HuggingFace format
  """
#   initialise_tracking()
  max_logging.log("loading hf model")
  hf_model = load_hf_model(config.model_name)
  max_logging.log("loading orbax model")
  training_state = load_model_state(config)
  max_logging.log("converting model")
  new_hf_model_params = convert_state_to_hf(training_state, config.model_name)
  max_logging.log(f"Saving HuggingFace model to path = {hf_model_path}")
  hf_model.save_pretrained(hf_model_path, state_dict=new_hf_model_params)



def main(argv: Sequence[str]):

#   jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
#   os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#   max_logging.log(argv)
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  max_logging.log("loading config")
  pyconfig.initialize(argv[:-1])
  max_logging.log("loaded config")
#   initialise_tracking(dir_prefix="/home/ljy/tracking")
  initialise_tracking()
  #Assuming the last argument is the path to save the converted checkpoint in HuggingFace format
  hf_model_path = argv[-1].split("=")[1]
  max_logging.log(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

  convert_orbax_hf(hf_model_path, pyconfig.config)

if __name__ == "__main__":
#   argv = sys.argv
#   main(argv)
  with jax.disable_jit():
    app.run(main)