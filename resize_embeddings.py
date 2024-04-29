# from EasyDel import AutoEasyDelModelForCausalLM
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import jax
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import upload_folder

"""
this in an example of how to update and resize model embedding size here we use pytorch to do that but you can use 
EasyDeL and jax to do the same thing but it will be a little bit harder than pytorch's so let just use pytorch for this
purpose and them load model in EasyDeL
"""


def add_special_tokens(model_name: str, num_new_tokens):
	# Add new special tokens to the tokenizer
	# num_new_tokens = len(new_tokens)

	# Determine the new vocabulary size
	new_vocab_size = num_new_tokens + 32000

	# Update the model's embedding layer and lm_head layer
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
	model.resize_token_embeddings(new_vocab_size)

	# Update the position embeddings (optional)
	if hasattr(model, "position_embeddings"):
		num_positions = model.position_embeddings.weight.size(0)
		new_positions = torch.arange(num_positions + num_new_tokens, device=model.device)
		model.position_embeddings.weight = torch.nn.Parameter(
			torch.cat((model.position_embeddings.weight,
					   torch.zeros(num_new_tokens, model.position_embeddings.embedding_dim)), dim=0)
		)

	return model


def main():
	model_name = "shared/Mixtral-8x22B-v0.1"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	print("loaded tokenizer")
	new_tokens = {"eos_token": "<|im_end|>", "additional_special_tokens": ["<|im_start|>",]}
	tokenizer.add_special_tokens(new_tokens)
	tokenizer.save_pretrained("./shared/Mixtral-8x22B-v0.1-resized-embeddings")
	print("saved tokenizer")
	print("loading model")
	model = add_special_tokens(model_name, 1024)
	print("resized embeddings")
	print("saving model")
	model.save_pretrained("./shared/Mixtral-8x22B-v0.1-resized-embeddings")
	print("saving tokenizer")
	# print("uploading")
	# upload_folder(repo_id = "a-normal-username/Mixtral-8x22B-v0.1-resized-embeddings",
	# 		   folder_path = "./shared/Mixtral-8x22B-v0.1-resized-embeddings")


if __name__ == "__main__":
	main()