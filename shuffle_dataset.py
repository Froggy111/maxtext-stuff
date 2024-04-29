from datasets import load_from_disk

train_dataset = load_from_disk("./shared/OpenHermes-2.5-tokenized-HF").shuffle().shuffle().shuffle()

train_dataset.save_to_disk("./shared/OpenHermes-2.5-tokenized-shuffled-HF")