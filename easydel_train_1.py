from EasyDel import (
    TrainArguments,
    AutoShardAndGatherFunctions,
    AutoEasyDelModelForCausalLM,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    SFTTrainer,
    CausalLanguageModelTrainer,
    MixtralConfig,
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
        ("model/embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        ("self_attn/o_proj/kernel", PartitionSpec(("sp", "fsdp"))),

        ("w1/kernel", PartitionSpec(("fsdp", "sp"))),
        ("w2/kernel", PartitionSpec(("fsdp", "sp"))),
        ("w3/kernel", PartitionSpec(("fsdp", "sp"))),
        ("gate/kernel", PartitionSpec(("fsdp", "sp"))),

        ("input_layernorm/kernel", PartitionSpec(None)),
        ("post_attention_layernorm/kernel", PartitionSpec(None)),

        ("model/norm/kernel", PartitionSpec(None)),
        ("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
        (".*", PartitionSpec(("fsdp", "sp"))),

    )
    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        dtype=jnp.bfloat16,
        auto_shard_params=True,
        partition_rules=rules,
        input_shape=(32, 512),
        query_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
        value_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
        key_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
        bias_partition_spec=PartitionSpec(("dp","fsdp"),None,"sp",None),
        attention_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
        generation_query_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", None, None),
        generation_bias_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", None, None),
    )

    print(params)

    print(model.config)

    model.config.get_partition_rules = lambda _: rules

    return model, params


huggingface_repo_id_or_path = "./shared/Mixtral-8x7B-v0.1-resized-embeddings"

train_dataset = load_from_disk("./shared/OpenHermes-2.5-tokenized_batch32")

train_dataset = train_dataset.remove_columns(
    [
        # "source", "model_name", "idx", "conversations", "hash", "skip_prompt_formatting",
        # "id", "language", "system_prompt", "category", "title", "custom_instruction",
        # "topic", "model", "views", "prompt", "avatarUrl"
        "prompt"
    ]
)

print(train_dataset[0])

model, params = load_model_sharded(huggingface_repo_id_or_path)

model.config.add_basic_configurations(
    attn_mechanism="flash",
    block_b=1,
    block_q=512,
    block_k=512,  # use 512 it's better since you are using max_sequence_length of 1024
    block_k_major=512,
    query_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
    value_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
    key_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
    bias_partition_spec=PartitionSpec(("dp","fsdp"),None,"sp",None),
    attention_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", "sp", None),
    generation_query_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", None, None),
    generation_bias_partition_spec=PartitionSpec(("dp", "fsdp"), "tp", None, None),
)

device_num = len(jax.devices())

original_max_position_embeddings = model.config.max_position_embeddings
model.config.freq_max_position_embeddings = model.config.max_position_embeddings
model.config.max_position_embeddings = 1024
model.config.c_max_position_embeddings = model.config.max_position_embeddings

max_length = 1024

print(model.config)

configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": jnp.bfloat16,
    "param_dtype": jnp.bfloat16,
    "input_shape": (32, 512)
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
    total_batch_size=32,
    max_training_steps=None,
    do_train=True,
    do_eval=False,
    backend="tpu",
    max_sequence_length=max_length,
    gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, -1, 1, 1),
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8,
    loss_re_mat="",
    dtype=jnp.bfloat16,
    save_dir="./shared/checkpoints",
    save_steps=50000,
    save_total_limit=10,
    truncation_mode="keep_end",
    init_input_shape=(32, 512),
    training_time="10000H"
)

trainer = CausalLanguageModelTrainer(
    arguments=train_arguments,
    dataset_train=train_dataset.shuffle(),
    dataset_eval=None,
    finetune=True,
)

output = trainer.train(flax.core.FrozenDict({"params": params}), state=None)
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")