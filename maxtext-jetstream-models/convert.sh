if [ "$fromFolder" = true ]
then
    export pathToDir=huggingface-models/${HFModelID}
fi

python convert.py --fromFolder ${fromFolder} \
    --pathToDir ${pathToDir} \
    --HFModelID ${HFModelID} \
    --revision ${revision}

python3 ${PWD}/maxtext/MaxText/llama_or_mistral_ckpt.py \
        --base-model-path ${PWD}/${pathToDir}/checkpoint.00.pth \
        --model-size ${MODEL_TYPE} \
        --maxtext-model-path ${PWD}/maxtext-models/${MODEL_NAME}

if [ "$MAKE_PARAM_ONLY" = true ]
then
    python3 ${PWD}/maxtext/MaxText/generate_param_only_checkpoint.py \
            ${PWD}/maxtext/MaxText/configs/base.yml \
            base_output_directory=${PWD}/maxtext-output \
            load_parameters_path=${PWD}/maxtext-models/${MODEL_NAME}/0/items \
            run_name=${PWD}/maxtext-models/${MODEL_NAME}-param-only \
            model_name=${MODEL_TYPE} \
            force_unroll=true
fi