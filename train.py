from EasyDel import (
	TrainArguments,
	CausalLanguageModelTrainer,
	AutoEasyDelModelForCausalLM,
	EasyDelOptimizers,
	EasyDelSchedulers,
	EasyDelGradientCheckPointers
)
from datasets import load_dataset
import flax, jax, torch
from jax import numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM

huggingface_repo_id_or_path = "./shared/Mixtral-8x22B-v0.1"

device_num = len(jax.devices())
model, params = AutoEasyDelModelForCausalLM.from_pretrained (
	huggingface_repo_id_or_path,
	device=jax.devices('cpu')[0],
	input_shape=(device_num, 1),
	device_map="auto",
	dtype=jnp.bfloat16,
	torch_dtype=torch.float16,
	)

model.config.add_basic_configurations(
    attn_mechanism="flash",  # flash , normal or splash (not fully supported yet on GPU,TPU) 
    block_b=1,
    block_q=512,
    block_k=512,
    block_k_major=512
)

model.config.original_max_position_embeddings = model.config.max_position_embeddings
model.config.freq_max_position_embeddings = model.config.max_position_embeddings
model.config.max_position_embeddings = 1024
model.config.c_max_position_embeddings = model.config.max_position_embeddings

max_length = 1024
tokenizer = AutoTokenizer.from_pretrained(
	huggingface_repo_id_or_path,
	trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
	"config": model.config,
	"dtype": jnp.bfloat16,
	"param_dtype": jnp.bfloat16,
	"input_shape": (device_num, model.config.block_q)
}

train_arguments = TrainArguments(
	model_class=type(model),
	model_name="my_first_model_to_train_using_easydel",
	num_train_epochs=3,
	configs_to_initialize_model_class=configs_to_initialize_model_class,
	learning_rate=5e-5,
	learning_rate_end=1e-6,
	optimizer=EasyDelOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
	scheduler=EasyDelSchedulers.LINEAR,
	# "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
	weight_decay=0.01,
	total_batch_size=64,
	max_training_steps=None,  # None to let trainer Decide
	do_train=True,
	do_eval=False,  # it's optional but supported 
	backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
	max_length=max_length,  # Note that you have to change this in the model config too
	gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
	sharding_array=(1, -1, 1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, 1, 1, -1)
	# everything training will be in sequence and model parallel automatic and share data between devices
	remove_ckpt_after_load=True,
	gradient_accumulation_steps=8,
	loss_re_mat="",
	dtype=jnp.bfloat16
)


def ultra_chat_prompting_process(data_chunk):
	prompt = ""

	for chunk in data_chunk:
		if prompt:
			prompt += "\n"
		if chunk["from"] == "system":
			prompt += f"<s>system\n{chunk['content']}</s>"
		elif chunk["from"] == "human":
			prompt += f"<s>user\n{chunk['content']}</s>"
		elif chunk["from"] == "gpt":
			prompt += f"<s>assistant\n{chunk['content']}</s>"

	return {"prompt": prompt}


tokenization_process = lambda data_chunk: tokenizer(
	data_chunk["prompt"],
	add_special_tokens=False,
	max_length=max_length,
	padding="max_length"
)

dataset = load_dataset("teknium/OpenHermes-2.5")
dataset_train = dataset["conversations"].map(ultra_chat_prompting_process, num_proc=12)
dataset_train = dataset_train.map(
	tokenization_process,
	num_proc=200,
	remove_columns=dataset_train.column_names
)

# you can do the same for evaluation process dataset

trainer = CausalLanguageModelTrainer(
	train_arguments,
	dataset_train,
	checkpoint_path=None
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")