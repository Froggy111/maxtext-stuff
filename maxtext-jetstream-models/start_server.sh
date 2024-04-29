#!/bin/bash

# export MODEL_NAME=mistral-7B-v0.1
export MODEL_NAME=OpenHermes-2.5-Mistral-7B
export MODEL_TYPE=mistral-7b
export RUN_NAME=server-test

export M_LOAD_PARAMETERS_PATH=${PWD}/shared/${MODEL_NAME}-param-only-checkpoint/
export M_PER_DEVICE_BATCH_SIZE=1
export M_MAX_PREFILL_PREDICT_LENGTH=1024
export M_MAX_TARGET_LENGTH=2048
export M_BASE_OUTPUT_DIRECTORY=${PWD}/maxtext-output
export M_MODEL_NAME="mistral-7b"
export M_ATTENTION="autoselected"
export M_PROMPT='''<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant'''
# export M_PROMPT="How have you been?"
export M_DECODE_SAMPLING_STRATEGY="greedy" # decode_sampling_strategy should be one of greedy, weighted, nucleus, or topk
export M_DECODE_SAMPLING_NUCLEUS_P=10 # set if you're doing nucleus / top-p
export M_DECODE_SAMPLING_TOP_K=0 # set if you're doing top-k
export M_DECODE_SAMPLING_TEMPERATURE=1
export TOKENIZER_PATH=${PWD}/shared/${MODEL_NAME}-param-only-checkpoint/tokenizer/tokenizer.model

# python ${PWD}/multiprocess_server.py \
#        ${PWD}/maxtext/MaxText/configs/base.yml \
#        # ${PWD}/shared/maxtext-jetstream-models/openhermes-2.5-config.yml \
#        tokenizer_path=${TOKENIZER_PATH} \
#        run_name=${RUN_NAME} \
#        steps=10 \
#        weight_dtype=bfloat16 \
#        async_checkpointing=false \
#        model_name=${MODEL_TYPE} \
#        scan_layers=false \
#        ici_fsdp_parallelism=1 \
#        ici_autoregressive_parallelism=-1 \

python ${PWD}/maxtext/MaxText/decode.py \
       ${PWD}/maxtext/MaxText/configs/base.yml \
       # ${PWD}/shared/maxtext-jetstream-models/openhermes-2.5-config.yml \
       tokenizer_path=${TOKENIZER_PATH} \
       run_name=${RUN_NAME} \
       steps=10 \
       weight_dtype=bfloat16 \
       async_checkpointing=false \
       model_name=${MODEL_TYPE} \
       scan_layers=false \
       ici_fsdp_parallelism=1 \
       ici_autoregressive_parallelism=-1 \