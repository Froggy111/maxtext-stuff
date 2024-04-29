from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

huggingface_repo_id_or_path = "./shared/Mixtral-8x22B-v0.1-resized-embeddings"

max_length = 1024
tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

sys_start = "<|im_start|>system\n"
user_start = "<|im_start|>user\n"
assistant_start = "<|im_start|>assistant\n"
in_between = "\n"

sys_start_ids = tokenizer(sys_start, padding = False, add_special_tokens = False, return_tensors = "np").input_ids[0]
user_start_ids = tokenizer(user_start, padding = False, add_special_tokens = False, return_tensors = "np").input_ids[0]
assistant_start_ids = tokenizer(assistant_start, padding = False, add_special_tokens = False, return_tensors = "np").input_ids[0]
in_between_ids = tokenizer(in_between, padding = False, add_special_tokens = False, return_tensors = "np").input_ids[0]

pad_token = 2

def process_dataset_label(data_chunk):
	prompt = ""
	input_ids = np.ones(1)
	attention_mask = np.ones(1)
	labels = np.ones(1)
	data_chunk = data_chunk['conversations']

	for chunk in data_chunk:
		if prompt:
			prompt += in_between
			input_ids = np.concatenate(input_ids, in_between_ids)
			labels = np.concatenate(labels, np.zeroes(in_between_ids.shape[0]))
			attention_mask = np.concatenate(attention_mask, np.ones(in_between_ids.shape[0]))
		if chunk["from"] == "system":
			to_label = False
			prompt += sys_start
			input_ids = np.concatenate(input_ids, sys_start_ids)
			labels = np.concatenate(labels, np.zeroes(sys_start_ids.shape[0]))
			attention_mask = np.concatenate(attention_mask, np.ones(sys_start_ids.shape[0]))
			to_add = f"{chunk['value']}<|im_end|>"
		elif chunk["from"] == "human":
			to_label = True
			prompt += user_start
			input_ids = np.concatenate(input_ids, user_start_ids)
			labels = np.concatenate(labels, np.zeroes(user_start_ids.shape[0]))
			attention_mask = np.concatenate(attention_mask, np.ones(user_start_ids.shape[0]))
			to_add = f"{chunk['value']}<|im_end|>"
		elif chunk["from"] == "gpt":
			to_label = False
			prompt += assistant_start
			input_ids = np.concatenate(input_ids, assistant_start_ids)
			labels = np.concatenate(labels, np.zeroes(assistant_start_ids.shape[0]))
			attention_mask = np.concatenate(attention_mask, np.ones(assistant_start_ids.shape[0]))
			to_add = f"{chunk['value']}<|im_end|>"
		
		prompt += to_add
		tokenizer_res = tokenizer(prompt, padding = False, add_special_tokens = False, return_tensors = "np")
		input_ids = np.concatenate(input_ids, tokenizer_res.input_ids[0])
		attention_mask = np.concatenate(input_ids, np.ones(tokenizer_res.input_ids[0].shape[0]))
		if to_label:
			labels = np.concatenate(labels, np.ones(tokenizer_res.input_ids[0].shape[0]))
		else:
			labels = np.concatenate(labels, np.zeroes(tokenizer_res.input_ids[0].shape[0]))
	
	attention_mask = np.concatenate(attention_mask, np.full(max_length - attention_mask.shape[0], pad_token))
	
	return {
		"prompt": prompt,
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"labels": labels,
	}

def process_dataset(data_chunk):
	prompt = ""

	for chunk in data_chunk['conversations']:
		if prompt:
			prompt += in_between
		if chunk["from"] == "system":
			prompt += sys_start
			prompt += f"{chunk['value']}<|im_end|>"
		elif chunk["from"] == "human":
			prompt += user_start
			prompt += f"{chunk['value']}<|im_end|>"
		elif chunk["from"] == "gpt":
			prompt += assistant_start
			prompt += f"{chunk['value']}<|im_end|>"
	
	tokenized_res = tokenizer (
		prompt,
		add_special_tokens=True,
		# truncation=True,
		# padding="max_length",
		# max_length=max_length,
		# return_overflowing_tokens=False,
		# return_length=False,
	)

	return {
		"prompt": prompt,
		"input_ids": tokenized_res.input_ids,
		# "attention_mask": tokenized_res.attention_mask,
	}
		

preprocessed_dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
original_columns = preprocessed_dataset.column_names

print("PROCESSING")
processed_dataset = preprocessed_dataset.map (
	process_dataset,
	remove_columns=original_columns,
	num_proc = 64,
)
print("PROCESSED")

print("SAVING")
processed_dataset.save_to_disk("./shared/OpenHermes-2.5-tokenized-unprocessed")
print("SAVED")