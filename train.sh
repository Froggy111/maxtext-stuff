python3 maxtext/MaxText/train.py \
	./shared/base.yml \
	load_parameters_path=gs://tpu-stuff-bucket/Mixtral-8x22B-v0.1-resized-embeddings-maxtext/0/items \
	run_name=Mixtral-8x22B-SFT-5 \
	per_device_batch_size=4 \
	model_name=mixtral-8x22b \
	ici_tensor_parallelism=1 \
	ici_fsdp_parallelism=64 \
	max_target_length=1024 \
	tokenizer_path=/home/ljy/shared/Mixtral-8x22B-v0.1-resized-embeddings \
	weight_dtype=bfloat16 \
	base_output_directory=gs://tpu-stuff-bucket/maxtext-runner-logs \
	dataset_path=/home/ljy/ \
	dataset_name=OpenHermes-2.5-tokenized-shuffled-HF