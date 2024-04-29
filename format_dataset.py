from datasets import load_from_disk

train_dataset = load_from_disk("./shared/OpenHermes-2.5-tokenized-unprocessed")
og_columns = train_dataset.column_names

def map_fn (batch):
	return {
		"inputs": batch["input_ids"],
		"targets": batch["input_ids"]
	}

train_dataset = train_dataset.map(map_fn, num_proc=64, remove_columns=og_columns)
train_dataset.save_to_disk("./shared/OpenHermes-2.5-tokenized-HF")
train_dataset = train_dataset.to_tf_dataset()
train_dataset.save("./shared/OpenHermes-2.5-tokenized-TFDS")