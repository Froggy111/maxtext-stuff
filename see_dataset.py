from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import copy
import time
import random

time.sleep(random.randint(1, 10))
train_dataset = load_from_disk("./shared/OpenHermes-2.5-tokenized-HF")
train_dataset = train_dataset.shuffle().shuffle().shuffle()
# eval_dataset = copy.deepcopy(train_dataset)
# # eval_dataset = load_from_disk("./shared/OpenHermes-2.5-tokenized-HF")
# eval_dataset = train_dataset.shuffle().shuffle().shuffle()
print("loaded")
# train_dataset = train_dataset.shard(8, 1)

# train_dataset = [
# 	{
# 		'prompt': """<|im_start|>user
# Every day, a tree drops 7 leaves. How many leaves would it drop in a month of February in a non-leap year? Include your logic.<|im_end|>
# <|im_start|>assistant
# Here's the logic behind this:

# 1. We know that February has 28 days in a non-leap year.
# 2. If the tree drops 7 leaves every day, then over the course of February, it would drop:
#    Leaves dropped in February = Leaves per day * Days in February
#    = 7 leaves * 28 days
#    = 196 leaves

# So, the tree would drop 196 leaves in February in a non-leap year.<|im_end|>"""
#     }
# ]

def print_map(batch):
	# print(batch)
	random_thing = batch["inputs"]
	return

train_dataset.map(print_map)
# print(np.array(train_dataset["inputs"][0]).shape)
# print(np.array(train_dataset["attention_mask"][0]).shape)
# print(train_dataset[0])

# huggingface_repo_id_or_path = "./shared/Mixtral-8x22B-v0.1-resized-embeddings"

# max_length = 1024
# tokenizer = AutoTokenizer.from_pretrained(
#     huggingface_repo_id_or_path,
#     trust_remote_code=True
# )
# tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer)
# print(train_dataset[0]['prompt'])
# print(tokenizer(train_dataset[0]['prompt']))