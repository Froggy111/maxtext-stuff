#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (i.e. https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar).

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_ASYNC_CHECKPOINTING=false
export JAX_PLATFORMS=cpu
# Download checkpoint, convert it to MaxText, and run inference
# pip3 install torch
# gsutil -m cp -r gs://maxtext-external/mixtral-8x7B-v0.1-Instruct /tmp
# python3 maxtext/MaxText/llama_or_mistral_ckpt.py --base-model-path /home/ljy/shared/Mixtral-8x22B-v0.1-resized-embeddings --model-size mixtral-8x22b --maxtext-model-path /home/ljy/shared/Mixtral-8x22B-v0.1-resized-embeddings-maxtext
python3 maxtext/MaxText/llama_or_mistral_ckpt.py \
	--base-model-path /home/ljy/shared/Mixtral-8x7B-v0.1-resized-embeddings \
	--model-size mixtral-8x7b \
	--maxtext-model-path gs://tpu-stuff-bucket/Mixtral-8x7B-v0.1-resized-embeddings-maxtext-2
# python3 maxtext/MaxText/decode.py MaxText/configs/base.yml load_parameters_path=./shared/Mixtral-8x22B-v0.1-maxtext run_name=runner_direct_${idx} per_device_batch_size=1 model_name=mixtral-8x22b tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral ici_tensor_parallelism=4 ici_fsdp_parallelism=16 max_prefill_predict_length=11 max_target_length=28 prompt="[INST] I love to [/INST]" autoregressive_decode_assert="That's great to hear! I love to learn new things and explore different interests" attention=dot_product

# Training
# python3 maxtext/MaxText/train.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-mixtral/test/${idx}/decode-ckpt-maxtext/0/items run_name=runner_${idx} per_device_batch_size=1 model_name=mixtral-8x7b ici_tensor_parallelism=4 ici_fsdp_parallelism=16 steps=10 max_target_length=1024 tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral
