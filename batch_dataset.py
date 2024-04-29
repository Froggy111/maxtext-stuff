from datasets import load_from_disk
from datasets import Dataset
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import math

dataset = load_from_disk("./shared/OpenHermes-2.5-EasyDel")
original_columns = dataset.column_names
batch_size = 32

def batch_fn (batch):
	# print(batch)
	prompts = [indiv["prompt"] for indiv in batch]
	input_ids = [indiv["input_ids"] for indiv in batch]
	attention_masks = [indiv["attention_mask"] for indiv in batch]
	# prompts = batch["prompt"]
	# input_ids = batch["input_ids"]
	# attention_masks = batch["attention_mask"]
	# print(len(prompts))
	# print(prompts, input_ids, attention_mask)
	prompts_batched = [prompts.pop(0)]
	input_ids_batched = np.expand_dims(np.array(input_ids.pop(0)), 0)
	attention_mask_batched = np.expand_dims(np.array(attention_masks.pop(0)), 0)
	for prompt, input_id, attention_mask in zip(prompts, input_ids, attention_masks):
		prompts_batched.append(prompt)
		input_ids_batched = np.concatenate([input_ids_batched, np.expand_dims(np.array(input_id), 0)], axis = 0)
		attention_mask_batched = np.concatenate([attention_mask_batched, np.expand_dims(np.array(attention_mask), 0)], axis = 0)
	
	out_dict = {
		"prompt": prompts_batched,
		"input_ids": input_ids_batched,
		"attention_mask": attention_mask_batched,
	}

	# print(len(out_dict["prompt"]))
	# print(out_dict["input_ids"].shape)
	# print(out_dict["attention_mask"].shape)

	if len(out_dict["prompt"]) < batch_size:
		return None
	# print(out_dict["input_ids"].shape)

	# print(out_dict)
	return out_dict

def batch_proc_fn (dataset):
	current_batch = []
	results = {
		"prompt": [],
		"input_ids": [],
		"attention_mask": [],
	}
	for prompt, input_ids, attention_mask in tqdm(zip(dataset["prompt"], dataset["input_ids"], dataset["attention_mask"])):
		# print(i)
		# print(indiv)
		current_batch.append (
			{
				"prompt": prompt,
				"input_ids": input_ids,
				"attention_mask": attention_mask,
			}
		)
		if len(current_batch) >= batch_size:
			processed = batch_fn(current_batch)
			current_batch = []
			results["prompt"].append(processed["prompt"])
			results["input_ids"].append(processed["input_ids"])
			results["attention_mask"].append(processed["attention_mask"])
		# print(len(results["prompt"]))
		# print(i)
		# if i >= end_idx - 1:
		# 	print("processing done")
		# 	break
	
	print("done processing")
	# q.put(results)
	print(results["prompt"][0])
	print(results["input_ids"][0])
	print(results["attention_mask"][0])
	return results

n_proc = 1
n_entries = len(dataset)

print(f"DATASET LENGTH: {n_entries}")

proclist = []
# for i in range(n_proc):
# 	q = mp.Queue(maxsize = 1000)
# 	print(math.floor((n_entries / n_proc) * i), math.ceil((n_entries / n_proc) * (i + 1)))
# 	# print(dataset[0])
# 	# print(dataset[math.floor((n_entries / n_proc) * i):math.ceil((n_entries / n_proc) * (i + 1))][0])
# 	p = mp.Process(
# 		target = batch_proc_fn,
# 		args = (dataset, math.floor((n_entries / n_proc) * i), math.ceil((n_entries / n_proc) * (i + 1)), q)
# 	)
# 	proclist.append(p)
# 	p.start()
# 	print(f"started process {i}")

# batched_dataset = {
# 	"prompt": [],
# 	"input_ids": [],
# 	"attention_mask": [],
# }

batched_dataset = batch_proc_fn(dataset)
batched_dataset = Dataset.from_dict(batched_dataset)
# for i, p in enumerate(proclist):
# 	p.join()
# 	print(f"finished process {i}")

# for i in range(n_proc):
# 	got_res = q.get(timeout=1)
# 	batched_dataset["prompt"].append(got_res["prompt"])
# 	batched_dataset["input_ids"].append(got_res["input_ids"])
# 	batched_dataset["attention_mask"].append(got_res["attention_mask"])

# dataset_stuff = dataset.map(
# 	batch_fn,
# 	batched = True,
# 	batch_size = batch_size,
# 	remove_columns = original_columns,
# 	num_proc = 32,
# 	# writer_batch_size=32,
# )

print(batched_dataset[0])

batched_dataset.save_to_disk(f"./shared/OpenHermes-2.5-tokenized-batch{batch_size}")