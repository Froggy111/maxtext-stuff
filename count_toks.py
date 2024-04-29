from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import tqdm

dataset = load_from_disk("./shared/OpenHermes-2.5-EasyDel")
print("LOADED FROM DISK")
print(dataset[0])

# print(dataset[:10])

# dataset = dataset['attention_mask']
# print("LOADED DATASET")

quantities = {}

for mask in tqdm.tqdm(dataset):
	# print(mask)
	length = np.count_nonzero(mask['attention_mask'])
	if length in quantities:
		quantities[length] += 1
	else:
		quantities[length] = 1

xpoints = np.array(list(quantities.keys()))
ypoints = np.array(list(quantities.values()))

toks_per_length = [quantities[key] * key for key in list(quantities.keys())]
total_toks = np.sum(toks_per_length)
total_sequences = np.sum(ypoints)

print(f"TOTAL TOKENS: {total_toks}")
print(f"AVERAGE TOKENS: {total_toks / total_sequences}")

plt.bar(xpoints, ypoints)
plt.show(block = True)
plt.savefig("./shared/plotted-newest.png")