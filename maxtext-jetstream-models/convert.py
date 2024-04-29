import torch
from transformers import AutoModelForCausalLM
import json
import argparse
import os, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--fromFolder", type=bool, default=False)
parser.add_argument("--HFModelID", type=str, required=False)
parser.add_argument("--revision", type=str, required=False)
parser.add_argument("--pathToDir", type=str, required=False)
parser.add_argument("--dtype", type=str, default="bf16")
args = parser.parse_args()

if args.dtype == "bf16":
	torch_dtype = torch.bfloat16
elif args.dtype == "fp16":
	torch_dtype = torch.float16
elif args.dtype == "fp32":
	torch_dtype = torch.float32
elif args.dtype == "int8":
	torch_dtype = torch.int8
elif args.dtype == "int4":
	torch_dtype = torch.int4
else:
	print("Invalid dtype. Must be one of bf16, fp16, fp32, int8, int4")
	exit()

if not args.fromFolder and not args.HFModelID:
	print("HFModelID required if not fromFolder")
	exit()
elif args.fromFolder and args.HFModelID:
	print("if loading from HF, turn off fromFolder")
	exit()
elif args.fromFolder and not args.pathToDir:
	print("pathToDir required if fromFolder")
	exit()

hf_model = True
hf_total_param_count = 0
compare = False
compare_ckpt_path = ""
compare_total_param_count = 0

param_config = {}

if not args.fromFolder and args.HFModelID:
	from huggingface_hub import HfApi
	api = HfApi()
	api.snapshot_download (
		repo_id = args.HFModelID,
		revision = args.revision if args.revision else "main",
		local_dir = os.join("huggingface-models", "args.HFModelID"),
		local_dir_use_symlinks = False
	)
	args.pathToDir = os.path.join("huggingface-models", args.HFModelID)


with open(os.path.join(args.pathToDir, "config.json"), "r") as f:
	loaded_param_config = json.load(f)
	param_config["dim"] = int(loaded_param_config["hidden_size"])
	param_config["n_layers"] = int(loaded_param_config["num_hidden_layers"])
	param_config["n_heads"] = int(loaded_param_config["num_attention_heads"])
	if "num_key_value_heads" in loaded_param_config:
		param_config["n_kv_heads"] = int(loaded_param_config["num_key_value_heads"])
	param_config["sliding_window"] = int(loaded_param_config["sliding_window"])
	param_config["vocab_size"] = int(loaded_param_config["vocab_size"])

model_layers = param_config["n_layers"]

def permute (
	w,
	n_heads = param_config["n_heads"],
	dim1 = param_config["dim"],
	dim2 = param_config["dim"]
):
	return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def unpermute (
	w,
	n_heads = param_config["n_heads"],
	dim1 = param_config["dim"],
	dim2 = param_config["dim"]
):
	w = w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
	w = w.transpose(1, 2)
	w = w.reshape(dim1, dim2)
	return w

if hf_model:
	model = AutoModelForCausalLM.from_pretrained(args.pathToDir, torch_dtype=torch_dtype)
	model = model.state_dict()
	model[f"tok_embeddings.weight"] = model.pop(f"model.embed_tokens.weight")
	model[f"norm.weight"] = model.pop(f"model.norm.weight")
	model[f"output.weight"] = model.pop(f"lm_head.weight")
	for i in range(model_layers):
		# model[f"layers.{i}.attention.wq.weight"] = model.pop(f"model.layers.{i}.self_attn.q_proj.weight")
		model[f"layers.{i}.attention.wq.weight"] = unpermute(model.pop(f"model.layers.{i}.self_attn.q_proj.weight"))
		# model[f"layers.{i}.attention.wk.weight"] = model.pop(f"model.layers.{i}.self_attn.k_proj.weight")
		model[f"layers.{i}.attention.wk.weight"] = unpermute(model.pop(f"model.layers.{i}.self_attn.k_proj.weight"),
															 n_heads = int(param_config["n_kv_heads"] if "n_kv_heads" in param_config else param_config["n_heads"]),
															 dim1 = int(param_config["dim"] / ((param_config["n_heads"]) / param_config["n_kv_heads"] if "n_kv_heads" in param_config else param_config["n_heads"])))
		model[f"layers.{i}.attention.wv.weight"] = model.pop(f"model.layers.{i}.self_attn.v_proj.weight")
		model[f"layers.{i}.attention.wo.weight"] = model.pop(f"model.layers.{i}.self_attn.o_proj.weight")
		model[f"layers.{i}.feed_forward.w1.weight"] = model.pop(f"model.layers.{i}.mlp.gate_proj.weight")
		model[f"layers.{i}.feed_forward.w2.weight"] = model.pop(f"model.layers.{i}.mlp.down_proj.weight")
		model[f"layers.{i}.feed_forward.w3.weight"] = model.pop(f"model.layers.{i}.mlp.up_proj.weight")
		model[f"layers.{i}.attention_norm.weight"] = model.pop(f"model.layers.{i}.input_layernorm.weight")
		model[f"layers.{i}.ffn_norm.weight"] = model.pop(f"model.layers.{i}.post_attention_layernorm.weight")
	
	for key in list(model.keys()):
		print(f"HF MODEL {key} SHAPE: {list(model[key].shape)}")
		key_param_count = 1
		for i in list(model[key].shape):
			key_param_count = key_param_count * i
		hf_total_param_count += key_param_count
	print(f"HF TOTAL PARAM COUNT: {hf_total_param_count / 1000000000}B")
	print(f"loaded model from {args.pathToDir}, saving to {args.pathToDir}/checkpoint.00.pth")
	torch.save(model.state_dict(), f"{args.pathToDir}/checkpoint.00.pth")
	print(f"saved model to {args.pathToDir}/checkpoint.00.pth successfully!")

if compare:
	torch_model = torch.load(compare_ckpt_path, map_location = 'cpu')
	for key in list(torch_model.keys()):
		# if "wq.weight" in key:
		#     torch_model[key] = permute(torch_model[key])
		# elif "wk.weight" in key:
		#     torch_model[key] = permute(torch_model[key], n_heads = 8, dim1 = 1024)
		print(f"COMPARE MODEL {key} SHAPE: {list(torch_model[key].shape)}")
		key_param_count = 1
		for i in list(model[key].shape):
			key_param_count = key_param_count * i
		compare_total_param_count += key_param_count
	print(f"COMPARE TOTAL PARAM COUNT: {hf_total_param_count / 1000000000}B")
	passed = True
	for key in list(torch_model.keys()):
		passed = torch.allclose(model[key], torch_model[key])
		print(f"COMPARE TENSOR EQUALITY of {key}: {torch.allclose(model[key], torch_model[key])}")
		print(f"(EQUAL PERCENTAGE) COMPARE TENSOR EQUALITY of {key}: {torch.sum(torch.eq(model[key], torch_model[key])).item()/torch_model[key].nelement() * 100}%")
		# print(f"COMPARE TENSOR EQUALITY of {key}: {torch.eq(model[key], torch_model[key])}")
	if passed:
		print("TEST PASSED")
	else:
		print("TEST FAILED")
	exit()