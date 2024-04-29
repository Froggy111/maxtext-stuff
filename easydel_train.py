from EasyDel import (
	TrainArguments,
	AutoShardAndGatherFunctions,
	AutoEasyDelModelForCausalLM,
	EasyDelOptimizers,
	EasyDelSchedulers,
	EasyDelGradientCheckPointers,
	SFTTrainer,
	CausalLanguageModelTrainer
)
from datasets import load_from_disk
import flax
import jax
from jax.sharding import PartitionSpec
from jax import numpy as jnp
from transformers import AutoTokenizer
import torch
import numpy as np

from jax_smi import initialise_tracking
initialise_tracking()

def load_model_sharded(pretrained_model_name_or_path: str):
	
	rules = (
		('model/embed_tokens/embedding', PartitionSpec("tp",('fsdp', 'sp'),)),
		('self_attn/(q_proj|k_proj|v_proj)/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('self_attn/o_proj/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('gate_proj/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('down_proj/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('up_proj/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('input_layernorm/kernel', PartitionSpec(None,)),
		('post_attention_layernorm/kernel', PartitionSpec(None,)),
		('model/norm/kernel', PartitionSpec(None,)),
		('lm_head/kernel', PartitionSpec(('fsdp', 'sp'),"tp")),
		('.*', PartitionSpec(('fsdp', 'sp'),))
	)

	print("getting sharding funcs")
	shard_fns, gather_fns = AutoShardAndGatherFunctions.from_pretrained(
		pretrained_model_name_or_path,
		rules
	)
	
	print("loading model")
	model, params = AutoEasyDelModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		shard_fns=shard_fns,
		torch_dtype = torch.bfloat16,
		dtype = jnp.bfloat16,
	)

	print(params)

	model.config.get_partition_rules = lambda _: rules

	return model, params

huggingface_repo_id_or_path = "./shared/Mixtral-8x7B-v0.1-resized-embeddings"

train_dataset = load_from_disk("./shared/OpenHermes-2.5-EasyDel-corrected-1024seq")

train_dataset = train_dataset.remove_columns (
	[
		"source", "model_name", "idx", "conversations", "hash", "skip_prompt_formatting",
		"id", "language", "system_prompt", "category", "title", "custom_instruction",
		"topic", "model", "views", "prompt", "avatarUrl"
	]
)

print(train_dataset[0])

model, params = load_model_sharded(huggingface_repo_id_or_path)

model.config.add_basic_configurations(
	attn_mechanism="flash",
	block_b=1,
	block_q=1024,
	block_k=1024,
	block_k_major=1024,
)

device_num = len(jax.devices())

original_max_position_embeddings = model.config.max_position_embeddings
model.config.freq_max_position_embeddings = model.config.max_position_embeddings
model.config.max_position_embeddings = 1024
model.config.c_max_position_embeddings = model.config.max_position_embeddings

configs_to_initialize_model_class = {
	"config": model.config,
	"dtype": jnp.bfloat16,
	"param_dtype": jnp.bfloat16,
	"input_shape": (device_num, model.config.block_q)
	# "input_shape": (1, model.config.block_q)
}

train_arguments = TrainArguments(
	model_class=type(model),
	model_name="SFT-EasyDeL",
	num_train_epochs=3,
	configs_to_initialize_model_class=configs_to_initialize_model_class,
	learning_rate=5e-5,
	learning_rate_end=1e-6,
	optimizer=EasyDelOptimizers.ADAMW,
	scheduler=EasyDelSchedulers.WARM_UP_COSINE,
	weight_decay=0.01,
	total_batch_size=1,
	max_training_steps=None,  # None to let trainer Decide
	do_train=True,
	do_eval=False,  # it's optional but supported 
	backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
	max_sequence_length=1024,  # Note that you have to change this in the model config too
	gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
	sharding_array=(1, 1, 1, -1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
	# everything training will be in sequence and model parallel automatic and share data between devices
	remove_ckpt_after_load=True,
	gradient_accumulation_steps=8,
	loss_re_mat="",
	dtype=jnp.bfloat16,
	save_dir="./shared/checkpoints",
	save_steps=50000,
	save_total_limit=10,
	truncation_mode = "keep_end"
)

# trainer = SFTTrainer(
#     arguments=train_arguments,
#     train_dataset=train_dataset.shuffle(),
#     eval_dataset=None,  # we don't have eval dataset rn :)
#     dataset_text_field=None,

#     packing=False,
#     num_of_sequences=1024,
# )

trainer = CausalLanguageModelTrainer (
	arguments = train_arguments,
	dataset_train = train_dataset.shuffle(),
	dataset_eval = None,
	finetune = True,
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")