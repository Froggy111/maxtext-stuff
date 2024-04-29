# python3 maxtext/MaxText/decode.py \
# 	shared/base.yml \
# 	load_parameters_path=gs://tpu-stuff-bucket/maxtext-runner-logs/Mixtral-8x22B-SFT-5/checkpoints/7850/items \
# 	run_name=test_run_random_followup_idk_what \
# 	per_device_batch_size=1 \
# 	model_name=mixtral-8x22b \
# 	tokenizer_path=./shared/Mixtral-8x22B-v0.1-resized-embeddings \
# 	ici_tensor_parallelism=1 \
# 	ici_fsdp_parallelism=4 \
# 	ici_autoregressive_parallelism=8 \
# 	max_prefill_predict_length=100 \
# 	max_target_length=512 \
# 	prompt='<|im_start|>system
# You are a helpful AI assistant. Think step by step and be concise in your working.<|im_end|>
# <|im_start|>user
# Write 10 sentences that end with apples.<|im_end|>
# <|im_start|>assistant' \
# 	attention=dot_product \
# 	# weight_dtype=bfloat16 \
python3 maxtext/MaxText/decode.py \
	shared/base.yml \
	load_parameters_path=gs://tpu-stuff-bucket/maxtext-runner-logs/Mixtral-8x22B-SFT-5/checkpoints/7850/items \
	run_name=test_run_random_followup_idk_what \
	per_device_batch_size=1 \
	model_name=mixtral-8x22b \
	tokenizer_path=./shared/Mixtral-8x22B-v0.1-resized-embeddings \
	ici_tensor_parallelism=1 \
	ici_fsdp_parallelism=4 \
	ici_autoregressive_parallelism=8 \
	max_prefill_predict_length=100 \
	max_target_length=512 \
	prompt='<|im_start|>user
Is a pound of steel or a kilogram of feathers heavier?<|im_end|>
<|im_start|>assistant' \
	attention=dot_product \
	# weight_dtype=bfloat16 \