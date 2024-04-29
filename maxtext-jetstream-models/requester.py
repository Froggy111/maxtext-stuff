'''
Implementation of continous batching with MaxEngine.
Code is based on JetStream's orchestrator.
Does not do tokenization. Tokenization is done by a middleman on a seperate thread to improve throughput.
Currently supports:
Async API (Server.request())
'''
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "maxtext/MaxText"))

import aiohttp.client_exceptions
import aiohttp.http_exceptions
import time, random, warnings, math, requests, json, httpx, \
	threading, queue, copy, traceback, signal, logging, asyncio, aiohttp

from multiprocessing import Queue as mpqueue
from multiprocessing import Process as mpprocess

from multiprocessing import set_start_method

from aiohttp import web
import uvicorn
from fastapi import FastAPI, HTTPException

import numpy as np

# SETTINGS

prompt = '''<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant'''

default_slot = {
	'active': False,
	'request_id': -1, # this is for use for the server
	'request_timestep': -1, # the timestep where the request was inserted. this is for use for the processor (-1 means to ignore)
	'request_config': {
		'stop_on_eos': False,
		'eos_token_id': 32000,
		'max_tokens': 2048,
		'max_input_tokens': 1024,
		'max_new_tokens': 2048,
		'clean_up_tokenization_spaces': True,
		'skip_special_tokens': False,
		# individual sampling is unused due to bad performance, all requests will follow the server_config sampling
		'sampling_strategy': 'greedy',
		'top_k': 40,
		'top_p': 0.9,
		'temperature': 0.6,
	},
	'input_tokens': [],
	'input_sequence': '',
	'output_tokens': [],
	'output_sequence': '', # unused for now
}

prompt = '''<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant'''

request_template = {
	'request_id': 1,
	'input_sequence': prompt,
	'config': {
		'stop_on_eos': False,
		'eos': 32000,
		'max_tokens': 2048,
		'max_input_tokens': 1024,
		'max_new_tokens': 1024,
		'clean_up_tokenization_spaces': True,
		'skip_special_tokens': False,
		'sampling_strategy': 'greedy',
		'top_k': 40,
		'top_p': 0.9,
		'temperature': 0.6,
	},
}

server_cfg = {
	# performance settings
	'DEBUG': True,
	'prefill_while_generating': True, # whether to allow prefill model pass while generating
	# IGNORE THE n-generation-cycles-wait. IT IS UNUSED.
	'n_generation_cycles_wait': 128, # number of generation cycles to wait before processing results.
	'n_generate_threads': 1, # number of engines/generate threads/processing threads. Recommended to keep at 1 and run seperate machines due to GIL contention.
	'n_tokenize_processes': 1, # total number of tokenizing processes. 1 should be enough.
	'n_detokenize_processes': 1, # total number of detokenizing processes. 1 should be enough.
	# IGNORE THE PREFILL THREAD SETTINGS. THEY ARE UNUSED
	'n_prefill_threads_per_engine': 1, # number of prefill threads per engine
	'prefill_batch_size': 1, # batch size when prefilling, NOT IMPLEMENTED YET, DO NOT MODIFY
	'prefill_max_store': 1, # how many prefill results to store per thread (at least 1)
	'prefill_request_get_timeout': 0.0000001, # timeout for getting a request from request queue to prefill in seconds

	'max_one_time_insertion': 384, # max number of insertions per generate cycle
	'insertion_timeout': 0.0000001, # timeout for recieving a next prefill result when inserting, in seconds
	'request_max_store': 1024, # max length of incoming request queue
	'response_max_store': 1024, # max length of outgoing response queue
	'sampled_tokens_max_store': 128, # max length of queue that stores sampled tokens waiting for processing, should be kept at 1
	
	# generation settings
	'stop_on_eos': False, # whether to stop when EOS on default
	'max_sequence_length': 2048,
	'max_prefill_length': 1024, # max prefill length
	'sampling_strategy': "greedy",
	'top_k': 40,
	'nucleus_top_p': 0.9,
	'temperature': 0.6,

	# api settings
	'server_host': '127.0.0.1',
	'server_port': 8080,
	'n_request_ids': 4096, # recommended to be large. Not too large though as it runs into memory errors for some reason.
	'max_concurrent_requests': 1024, # should be at least larger than combined total batch size! should be smaller than request/response store queue
	'delay_before_reject': 1, # seconds to delay before rejecting request to prevent spamming from client side

	# tokenizer settings
	'tokenizer_config': {
		'path': "/home/ljy/tokenizer",
		'use_fast': True,
		'padding_side': "right",
		'pad_token_id': 2,
		'bos_token_id': 1,
		'eos_token_id': 3,
		'unk_token_id': 0,
		'possible_lengths': [16, 32, 64, 128, 256, 512,
					   1024, 2048, 4096, 8192, 16384, 32768,],
	},
}

def main():
	# request_template['config']['stop_on_eos'] = True
	test_request = copy.deepcopy(request_template)
	n_to_run = 384 * 1024
	n_per_batch = 384 * 16
	n_prefill_toks = 1024
	n_generate_toks = 1024
	n_concurrent = 384 * 2
	# current_concurrent = 0
	# total_genned = 0

	def make_request(request):
		print("REQUEST: ", request)
		while True:
			try:
				thing = requests.get(f'http://{server_cfg["server_host"]}:{server_cfg["server_port"]}/request', params = {'request': json.dumps(request)})
				# thing = asyncio.run(thing)
				thing = thing.json()
				print(thing)
			except requests.exceptions.HTTPError as e:
				print(e)

	async def async_request(request):
		print("REQUEST: ", request)
		while True:
			# try:
			# 	thing = await httpx.get(f'http://{server_cfg["server_host"]}:{server_cfg["server_port"]}/request', params = {'request': json.dumps(request)})
			# 	# thing = asyncio.run(thing)
			# 	thing = thing.json()
			# 	print(thing)
			# except httpx.exceptions.HTTPError as e:
			# 	print(e)
			try:
				async with aiohttp.ClientSession() as session:
					async with session.get(f'http://{server_cfg["server_host"]}:{server_cfg["server_port"]}/request', params = {'request': json.dumps(request)}) as response:
						thing = await response.json()
						print(thing)
			except aiohttp.http_exceptions.HttpProcessingError as e:
				print(e)
	async def async_request_manager(request):
		coros = [async_request(copy.deepcopy(request), ) for _ in range(n_concurrent)]
		await asyncio.gather(*coros)
	# time.sleep(10000000)
	logging.info("STARTING BENCHMARK")
	tstart = time.perf_counter()
	# for i in range(int(n_to_run / n_per_batch)):
	# 	asyncio.run(test_all_requesting())
	# asyncio.run(cont_request(test_request))
	# asyncio.run(cont_request_manager(test_request))
	# make_request(test_request)
	asyncio.run(async_request_manager(test_request))
	tend = time.perf_counter()
	queries_per_second = n_to_run / (tend - tstart)
	prefill_toks_per_second = n_prefill_toks * n_to_run / (tend - tstart)
	generate_toks_per_second = n_generate_toks * n_to_run / (tend - tstart)
	logging.info(f'Queries per second: {queries_per_second}')
	logging.info(f'Prefill tokens per second: {prefill_toks_per_second}')
	logging.info(f'Generate tokens per second: {generate_toks_per_second}')
	with open('results.txt', 'w') as f:
		f.write(f'Queries per second: {queries_per_second}\n')
		f.write(f'Prefill tokens per second: {prefill_toks_per_second}\n')
		f.write(f'Generate tokens per second: {generate_toks_per_second}\n')
	traceback.print_exc()
	os.kill(os.getpid(), signal.SIGKILL)

main()