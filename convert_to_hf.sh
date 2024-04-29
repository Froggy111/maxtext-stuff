export JAX_PLATFORMS=cpu
python maxtext/MaxText/convert_to_hf.py \
	./shared/base.yml \
	load_parameters_path=gs://tpu-stuff-bucket/maxtext-runner-logs/Mixtral-8x22B-SFT-5/checkpoints/5900/items \
	model_name=mixtral-8x22b \
	ici_tensor_parallelism=1 \
	ici_fsdp_parallelism=1 \
	weight_dtype=bfloat16 \
	path_of_hf_model=/home/ljy/shared/Mixtral-8x22B-OpenHermes-3rd-epoch