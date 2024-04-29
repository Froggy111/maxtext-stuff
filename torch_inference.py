from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from accelerate import init_empty_weights

# model_name="shared/Mixtral-8x22B-OpenHermes-4th-epoch"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# other_tokenizer = AutoTokenizer.from_pretrained("shared/Mixtral-8x22B-v0.1-resized-embeddings", trust_remote_code=True)
other_other_tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", trust_remote_code=True)
# preemptible-1 is 3rd epoch, non-preemptible-2 is 4th epoch
# prompt = """<|im_start|>system
# You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
# <|im_start|>user
# Hello, who are you?<|im_end|>
# <|im_start|>assistant"""

# inputs = tokenizer([prompt], return_tensors="pt")
# inputs_2 = other_tokenizer([prompt], return_tensors="pt")
# inputs_3 = other_other_tokenizer([prompt], return_tensors="pt")
# print(inputs)
# print(inputs_2)
# print(inputs_3)
# assert torch.equal(inputs.input_ids,inputs_2.input_ids)
# assert torch.equal(inputs_3.input_ids,inputs_2.input_ids)
# decoded = tokenizer.decode(inputs.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# decoded_2 = other_tokenizer.decode(inputs.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# decoded_3 = other_other_tokenizer.decode(inputs.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# print(decoded)
# print(decoded_2)
# print(decoded_3)
# decoded = tokenizer.decode(inputs_3.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# decoded_2 = other_tokenizer.decode(inputs_3.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# decoded_3 = other_other_tokenizer.decode(inputs_3.input_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=False)
# print(decoded)
# print(decoded_2)
# print(decoded_3)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)


# streamer = TextStreamer(tokenizer)

# _ = model.generate(**inputs, streamer=streamer, max_new_tokens=100)

other_other_tokenizer.save_pretrained("shared/Mixtral-8x22B-OpenHermes-3rd-epoch")