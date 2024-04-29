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

r"""Convert weights from a Llama or Mistral model to a MaxText one.

Usage:

Get LLaMA pytorch_vars from Meta

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
	--maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

For large size model (e.g. 70B model), this script requires large memory VM.
The script load and save weights in a single pass.
To fit less memory, modify convert() to load/save weights in multiple passes.
Each pass, load and save partial weights (subset of all weight variables).
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib

import numpy as np

import checkpointing
import jax
from jax import numpy as jnp
from flax.training import train_state
import max_logging
from train import save_checkpoint
import torch
import sys
import os
from safetensors.torch import load_file
import logging
import tqdm
import gc

logging.critical("started logging")

logger = logging.getLogger("converter")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler('/home/ljy/out.txt'))

def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jnp.concatenate((evens, odds), axis=arr.ndim-1)


MODEL_PARAMS_DICT = {
	'llama2-70b': {
		'num_layers': 80,
		'num_heads': 64,
		'num_kv_heads': 8,
		'dims_per_head': 128,
		'vocab': 32128,
	},
	'llama2-13b': {
		'num_layers': 40,
		'num_heads': 40,
		'num_kv_heads': 40,
		'dims_per_head': 128,
		'vocab': 32128,
	},
	'llama2-7b': {
		'num_layers': 32,
		'num_heads': 32,
		'num_kv_heads': 32,
		'dims_per_head': 128,
		'vocab': 32128,
	},
	'mistral-7b': {
		'num_layers': 32,
		'num_heads': 32,
		'num_kv_heads': 8,
		'dims_per_head': 128,
		'vocab': 32000,
		'base_emb_dim': 4096,
		'base_mlp_dim': 14336,
	},
	'mixtral-8x7b': {
		'num_layers': 32,
		'num_heads': 32,
		'num_kv_heads': 8,
		'dims_per_head': 128,
		'vocab': 32128,
		'base_emb_dim': 4096,
		'base_mlp_dim': 14336,
		'num_experts': 8,
	},
	'mixtral-8x22b': {
		'num_layers': 56,
		'num_heads': 48,
		'num_kv_heads': 8,
		'dims_per_head': 128,
		'vocab': 32128,
		'base_emb_dim': 6144,
		'base_mlp_dim': 16384,
		'num_experts': 8
	}
}

SIMULATED_CPU_DEVICES_COUNT = 1

def unpermute (
	w,
	n_heads,
	dim1,
	dim2
):
	w = w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
	w = w.transpose(1, 2)
	w = w.reshape(dim1, dim2)
	return w

def convert(base_model_path, maxtext_model_path, model_size):
	"""
	Function to convert the checkpoint at base_model_path into Orbax checkpoint
	for MaxText and save at maxtext_model_path

	Attributes:
	base_model_path: checkpoint path
	maxtext_model_path: Path to save the MaxText checkpoint to
	model_size: llama2-7b to 70b, mistral-7b, or mixtral-8x7b
	"""
	"""Convert model to maxtext."""
	model_params = MODEL_PARAMS_DICT[model_size]
	base_num_decoder_layers = model_params['num_layers']
	base_num_query_heads = model_params['num_heads']
	head_dim = model_params['dims_per_head']
	base_num_kv_heads = model_params['num_kv_heads']
	vocab_size = model_params['vocab']
	num_experts = model_params['num_experts'] if 'num_experts' in model_params else None

	mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
	s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))  # shards first axis
	s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))  # shards second axis
	s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))  # no sharding
	device_num = len(jax.devices())

	def checkpoint_device_put(arr):
		def cb(i):
			return arr[i]
		if arr.shape[0] % device_num == 0:
			print("sharding first axis")
			# return_arr = jax.device_put(arr, device=s1)
			return_arr = jax.make_array_from_callback(arr.shape, s1, cb)
		elif len(arr.shape) > 1 and arr.shape[1] % device_num == 0:
			print("sharding second axis")
			# return_arr = jax.device_put(arr, device=s2)
			return_arr = jax.make_array_from_callback(arr.shape, s2, cb)
		else:
			print("no sharding was possible, replicating")
			# return_arr = jax.device_put(arr, device=s3)
			return_arr = jax.make_array_from_callback(arr.shape, s3, cb)
		print(type(return_arr))
		print(return_arr.shape)
		return return_arr

	while True:
		try:
			logger.info(f'Loading the base model from {base_model_path}')
			# Skip any hidden files for checkpoints
			ckpt_paths = sorted(pathlib.Path(base_model_path).glob('*.safetensors'))
			pytorch_vars = {}
			for i, ckpt_path in enumerate(ckpt_paths):
				logger.info(f'Loading checkpoint {i+1} of {len(ckpt_paths)} ...')
				checkpoint = load_file(ckpt_path)
				# checkpoint = torch.load(ckpt_path, map_location='cpu')
				for k in checkpoint.keys():
					pytorch_vars[k] = checkpoint[k]
				# pytorch_vars[i] = checkpoint
			# pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]
			pytorch_vars = [pytorch_vars]
			# print([torch_vars.keys() for torch_vars in pytorch_vars])
			break
		except Exception as e:
			print(e)

	layer_key = 'gate' if num_experts else 'mlp'
	jax_weights = {
		'decoder': {
			'layers': {
				layer_key: {},
				'pre_self_attention_layer_norm': {},
				'post_self_attention_layer_norm': {},
				'self_attention': {},
			},
			'decoder_norm': {
				'scale': checkpoint_device_put(jnp.array(pytorch_vars[0].pop('model.norm.weight').float().numpy(), dtype = jnp.bfloat16))
			},
			'logits_dense': {
				'kernel': checkpoint_device_put(jnp.concatenate([jnp.array(var.pop('lm_head.weight').float().numpy(), dtype = jnp.bfloat16)
											for var in pytorch_vars], axis=0).transpose()[:, :vocab_size])
			}
		},
		'token_embedder': {
			'embedding': checkpoint_device_put(jnp.concatenate([jnp.array(var.pop('model.embed_tokens.weight').float().numpy(), dtype = jnp.bfloat16)
											for var in pytorch_vars], axis=1)[:vocab_size, :])

		}

	}

	layer_weight = {
		'pre_self_attention_layer_norm': {
			'scale': []
		},
		'post_self_attention_layer_norm': {
			'scale': []
		}
	}

	if num_experts is None:
		layer_weight['mlp'] = {
			'wi_0': {
				'kernel': []
			},
			'wi_1': {
				'kernel': []
			},
			'wo': {
				'kernel': []
			},
		}
	else:
		layer_weight['gate'] = {
				'kernel': []
			}

		for k in range(num_experts):
			jax_weights['decoder']['layers'][f'mlp_{k}'] = {}
			layer_weight[f'mlp_{k}'] = {
				'wi_0': {
					'kernel': []
				},
				'wi_1': {
					'kernel': []
				},
				'wo': {
					'kernel': []
				},
			}

	self_attention = {
		'query': {
			'kernel': []
		},
		'key': {
			'kernel': []
		},
		'value': {
			'kernel': []
		},
		'out': {
			'kernel': []
		},
	}

	with jax.default_device(jax.devices('cpu')[0]):
		for layer_idx in tqdm.tqdm(range(base_num_decoder_layers)):
			gc.collect()
			logger.info(f"processing layer {layer_idx}")
			logger.info("processing attention")
			wq = jnp.concatenate (
				[
					jnp.array(
						unpermute(
							var.pop(f'model.layers.{layer_idx}.self_attn.q_proj.weight'),
							n_heads=base_num_query_heads,
							dim1=base_num_query_heads * head_dim,
							dim2=base_num_query_heads * head_dim,
						).float().numpy(),
						dtype = jnp.bfloat16
					) for var in pytorch_vars
				], axis=0).transpose()
			wk = jnp.concatenate (
				[
					jnp.array(
						unpermute(
							var.pop(f'model.layers.{layer_idx}.self_attn.k_proj.weight'),
							n_heads=base_num_kv_heads,
							dim1=base_num_kv_heads * head_dim,
							dim2=base_num_query_heads * head_dim,
						).float().numpy(),
						dtype = jnp.bfloat16
					) for var in pytorch_vars
				], axis=0).transpose()
			wv = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.self_attn.v_proj.weight').float().numpy(), dtype = jnp.bfloat16)
								for var in pytorch_vars], axis=0).transpose()

			wq = jnp.reshape(wq, [base_num_query_heads * head_dim,
							base_num_query_heads, head_dim])
			wk = jnp.reshape(wk, [base_num_query_heads * head_dim,
							base_num_kv_heads, head_dim])
			wv = jnp.reshape(wv, [base_num_query_heads * head_dim,
							base_num_kv_heads, head_dim])
			wq = permute_to_match_maxtext_rope(wq)
			# print(head_dim)
			wk = permute_to_match_maxtext_rope(wk)

			w_post = jnp.concatenate(
				[
					jnp.array(var.pop(f'model.layers.{layer_idx}.self_attn.o_proj.weight').float().numpy(), dtype = jnp.bfloat16)
					for var in pytorch_vars
				],
				axis=1,
			)

			w_post = jnp.reshape(
				w_post, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

			self_attention['query']['kernel'].append(wq)
			self_attention['key']['kernel'].append(wk)
			self_attention['value']['kernel'].append(wv)
			self_attention['out']['kernel'].append(w_post)
			pre_self_attention_layernorm = jnp.array(pytorch_vars[0].pop(f'model.layers.{layer_idx}.input_layernorm.weight').type(
				torch.float16).numpy(), dtype = jnp.bfloat16)
			post_self_attention_layernorm = jnp.array(pytorch_vars[0].pop(f'model.layers.{layer_idx}.post_attention_layernorm.weight').type(
				torch.float16).numpy(), dtype = jnp.bfloat16)
			layer_weight['pre_self_attention_layer_norm']['scale'].append(
				pre_self_attention_layernorm)
			layer_weight['post_self_attention_layer_norm']['scale'].append(
				post_self_attention_layernorm)

			logger.info("processing MLP")
			if num_experts is None:
				wi_0 = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.mlp.gate_proj.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=0).transpose()
				wi_1 = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.mlp.up_proj.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=0).transpose()
				wo = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.mlp.down_proj.weight').float().numpy(), dtype = jnp.bfloat16)
									for var in pytorch_vars], axis=1).transpose()
				layer_weight['mlp']['wi_0']['kernel'].append(wi_0)
				layer_weight['mlp']['wi_1']['kernel'].append(wi_1)
				layer_weight['mlp']['wo']['kernel'].append(wo)
			else:
				gate = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.block_sparse_moe.gate.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=0).transpose()
				layer_weight['gate']['kernel'].append(gate)
				for k in tqdm.tqdm(range(num_experts)):
					wi_0 = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.block_sparse_moe.experts.{k}.w1.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=0).transpose()
					wi_1 = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.block_sparse_moe.experts.{k}.w3.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=0).transpose()
					wo = jnp.concatenate([jnp.array(var.pop(f'model.layers.{layer_idx}.block_sparse_moe.experts.{k}.w2.weight').float().numpy(), dtype = jnp.bfloat16)
										for var in pytorch_vars], axis=1).transpose()
					layer_weight[f'mlp_{k}']['wi_0']['kernel'].append(wi_0)
					layer_weight[f'mlp_{k}']['wi_1']['kernel'].append(wi_1)
					layer_weight[f'mlp_{k}']['wo']['kernel'].append(wo)

	# layers, base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
	# base_num_query_heads, layers,head_dim, base_num_query_heads * head_dim

	logger.info("processing self attention query kernel")
	self_attention['query']['kernel'] = jnp.array(
		self_attention['query']['kernel'])
	self_attention['query']['kernel'] = jnp.transpose(
		self_attention['query']['kernel'], axes=(1, 0, 2, 3))
	
	logger.info("processing self attention key kernel")
	self_attention['key']['kernel'] = jnp.array(
		self_attention['key']['kernel'])
	self_attention['key']['kernel'] = jnp.transpose(
		self_attention['key']['kernel'], axes=(1, 0, 2, 3))
	
	logger.info("processing self attention value kernel")
	self_attention['value']['kernel'] = jnp.array(
		self_attention['value']['kernel'])
	self_attention['value']['kernel'] = jnp.transpose(
		self_attention['value']['kernel'], axes=(1, 0, 2, 3))
	self_attention['value']['kernel'] = checkpoint_device_put(self_attention['value']['kernel'])
	
	logger.info("processing self attention out kernel")
	self_attention['out']['kernel'] = jnp.array(self_attention['out']['kernel'])
	self_attention['out']['kernel'] = jnp.transpose(
		self_attention['out']['kernel'], axes=(2, 0, 3, 1))
	self_attention['out']['kernel'] = checkpoint_device_put(self_attention['out']['kernel'])

	# scale the query weights
	logger.info("scaling query weights")
	self_attention['query']['kernel'] = self_attention['query']['kernel'] / \
		jnp.sqrt(head_dim)
	self_attention['query']['kernel'] = checkpoint_device_put(self_attention['query']['kernel'])

	jax_weights['decoder']['layers']['self_attention'] = self_attention

	# self attention layer norm and swap the layer index
	logger.info("processing pre attention layernorm")
	layer_weight['pre_self_attention_layer_norm']['scale'] = jnp.array(
		layer_weight['pre_self_attention_layer_norm']['scale'])
	layer_weight['pre_self_attention_layer_norm']['scale'] = jnp.transpose(
		layer_weight['pre_self_attention_layer_norm']['scale'],
		axes=(1, 0))
	layer_weight['pre_self_attention_layer_norm']['scale'] = checkpoint_device_put(layer_weight['pre_self_attention_layer_norm']['scale'])

	logger.info("processing post attention layernorm")
	layer_weight['post_self_attention_layer_norm']['scale'] = jnp.array(
		layer_weight['post_self_attention_layer_norm']['scale'])
	layer_weight['post_self_attention_layer_norm']['scale'] = jnp.transpose(
		layer_weight['post_self_attention_layer_norm']['scale'],
		axes=(1, 0))
	layer_weight['post_self_attention_layer_norm']['scale'] = checkpoint_device_put(layer_weight['post_self_attention_layer_norm']['scale'])

	jax_weights['decoder']['layers']['pre_self_attention_layer_norm'] = layer_weight['pre_self_attention_layer_norm']
	jax_weights['decoder']['layers']['post_self_attention_layer_norm'] = layer_weight['post_self_attention_layer_norm']

	if num_experts is None:
		layer_weight['mlp']['wi_0']['kernel'] = jnp.array(
			layer_weight['mlp']['wi_0']['kernel'])
		layer_weight['mlp']['wi_1']['kernel'] = jnp.array(
			layer_weight['mlp']['wi_1']['kernel'])
		layer_weight['mlp']['wo']['kernel'] = jnp.array(
			layer_weight['mlp']['wo']['kernel'])
		# swap the layer index
		layer_weight['mlp']['wi_0']['kernel'] = jnp.transpose(
			layer_weight['mlp']['wi_0']['kernel'], axes=(1, 0, 2))
		layer_weight['mlp']['wi_1']['kernel'] = jnp.transpose(
			layer_weight['mlp']['wi_1']['kernel'], axes=(1, 0, 2))
		layer_weight['mlp']['wo']['kernel'] = jnp.transpose(
			layer_weight['mlp']['wo']['kernel'], axes=(1, 0, 2))

		jax_weights['decoder']['layers']['mlp'] = layer_weight['mlp']
	else:
		logger.info("processing gate")
		layer_weight['gate']['kernel'] = jnp.array(layer_weight['gate']['kernel'])
		layer_weight['gate']['kernel'] = jnp.transpose(
			layer_weight['gate']['kernel'], axes=(1, 0, 2))
		layer_weight['gate']['kernel'] = checkpoint_device_put(layer_weight['gate']['kernel'])

		jax_weights['decoder']['layers']['gate'] = layer_weight['gate']
		for k in tqdm.tqdm(range(num_experts)):
			logger.info(f"processing expert {k}")
			
			logger.info("processing wi_0 kernel")
			layer_weight[f'mlp_{k}']['wi_0']['kernel'] = jnp.array(
				layer_weight[f'mlp_{k}']['wi_0']['kernel'])
			layer_weight[f'mlp_{k}']['wi_0']['kernel'] = jnp.transpose(
				layer_weight[f'mlp_{k}']['wi_0']['kernel'], axes=(1, 0, 2))
			layer_weight[f'mlp_{k}']['wi_0']['kernel'] = checkpoint_device_put(layer_weight[f'mlp_{k}']['wi_0']['kernel'])
			
			logger.info("processing wi_1 kernel")
			layer_weight[f'mlp_{k}']['wi_1']['kernel'] = jnp.array(
				layer_weight[f'mlp_{k}']['wi_1']['kernel'])
			layer_weight[f'mlp_{k}']['wi_1']['kernel'] = jnp.transpose(
				layer_weight[f'mlp_{k}']['wi_1']['kernel'], axes=(1, 0, 2))
			layer_weight[f'mlp_{k}']['wi_1']['kernel'] = checkpoint_device_put(layer_weight[f'mlp_{k}']['wi_1']['kernel'])
			
			logger.info("processing wo kernel")
			layer_weight[f'mlp_{k}']['wo']['kernel'] = jnp.array(
				layer_weight[f'mlp_{k}']['wo']['kernel'])
			layer_weight[f'mlp_{k}']['wo']['kernel'] = jnp.transpose(
				layer_weight[f'mlp_{k}']['wo']['kernel'], axes=(1, 0, 2))
			layer_weight[f'mlp_{k}']['wo']['kernel'] = checkpoint_device_put(layer_weight[f'mlp_{k}']['wo']['kernel'])

			jax_weights['decoder']['layers'][f'mlp_{k}'] = layer_weight[f'mlp_{k}']

	# dummy configs for the checkpoint_manager
	step_number_to_save_new_ckpt = 0
	enable_checkpointing = True
	async_checkpointing = False
	save_interval_steps = 1

	logger.info("creating checkpoint manager")
	checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
		maxtext_model_path,
		enable_checkpointing,
		async_checkpointing,
		save_interval_steps
	)

	logger.info("creating trainstate")
	state_new = train_state.TrainState(
		step=0,
		apply_fn=None,
		params={'params': jax_weights},
		tx=None,  # type: ignore
		opt_state={}
	)
	
	logger.info("saving checkpoint")
	if checkpoint_manager is not None:
		if save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
			max_logging.log(
				f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
		# Upon preemption, exit when and only when all ongoing saves are complete.
		if checkpoint_manager.reached_preemption(0):
			checkpoint_manager.wait_until_finished()
			sys.exit()


if __name__ == '__main__':
	from jax_smi import initialise_tracking
	initialise_tracking()
	jax.distributed.initialize()
	parser = argparse.ArgumentParser()
	parser.add_argument('--base-model-path', type=str, required=True)
	parser.add_argument('--maxtext-model-path', type=str, required=True)
	parser.add_argument('--model-size', type=str, required=True)

	args = parser.parse_args()

	if args.model_size not in MODEL_PARAMS_DICT:
		raise NotImplementedError

	os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}'

	convert(args.base_model_path, args.maxtext_model_path, args.model_size)
