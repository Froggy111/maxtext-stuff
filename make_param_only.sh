python3 maxtext/MaxText/generate_param_only_checkpoint.py \
        shared/base.yml \
		ici_fsdp_parallelism=-1 \
        base_output_directory=gs://tpu-stuff-bucket/maxtext-runner-logs/ \
        load_parameters_path=gs://tpu-stuff-bucket/maxtext-runner-logs/Mixtral-8x22B-SFT-5/checkpoints/1800/items \
        run_name=gs://tpu-stuff-bucket/maxtext-runner-logs/Mixtral-8x22B-SFT-5-decode-only/checkpoints/1800 \
        model_name=mixtral-8x22b \
        force_unroll=true