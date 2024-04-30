'''
Implementation of continous batching with MaxEngine.
Code is based on JetStream's orchestrator.
Does not do tokenization. Tokenization is done by a middleman on a seperate thread to improve throughput.
Currently supports:
Async API (Server.request())
'''
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "/home/ljy/maxtext-stuff/maxtext-jetstream-models/maxtext/MaxText"))

import aiohttp.http_exceptions
import jax, transformers, time, random, warnings, multiprocess, math, requests, json, \
	threading, queue, copy, traceback, signal, logging, asyncio, aiohttp, multiprocessing

from multiprocessing import Queue as mpqueue
from multiprocessing import Process as mpprocess

from multiprocessing import set_start_method

from aiohttp import web
import uvicorn
from fastapi import FastAPI, HTTPException

import numpy as np
from jax import numpy as jnp

from jax_smi import initialise_tracking

import maxengine
# from ..maxtext.MaxText import maxengine

from ..maxtext.MaxText import pyconfig

from jetstream.engine.token_utils import take_nearest_length

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
		'path': "teknium/OpenHermes-2.5-Mistral-7B",
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

def main(engine_configs, server_config):
	# server_process = multiprocessing.Process (
	# 	target = start_server,
	# 	args = (engine_configs, server_config),
	# )
	# server_process.daemon = True
	# server_process.start()
	server = Server (
		engine_configs = engine_configs,
		server_config = server_config,
	)
	server.start()
	server.start_fastapi_server()
	request_template['config']['stop_on_eos'] = True
	test_request = copy.deepcopy(request_template)
	n_to_run = 384 * 1024
	n_per_batch = 384 * 16
	n_prefill_toks = 1024
	n_generate_toks = 1024
	n_concurrent = 384 * 2
	# current_concurrent = 0
	# total_genned = 0

	async def cont_request_manager(request):
		coros = [cont_request(copy.deepcopy(request), ) for _ in range(n_concurrent)]
		await asyncio.gather(*coros)
	async def cont_request(request):
		# nonlocal current_concurrent
		# nonlocal total_genned
		requester_total = 0
		while requester_total < int(n_to_run / n_concurrent):
		# while True:
			# logging.info(requester_total)
			# if current_concurrent >= n_concurrent:
			# 	await asyncio.sleep(1)
			# 	continue
			# current_concurrent += 1
			request['config']['max_new_tokens'] = random.choice([
				16, 32, 64, 128, 256, 512, 1024,
				2048, 4096, 8192, 16384, 32768,])
			request['config']['stop_on_eos'] = random.choice([True, False])
			resp = await server.requester(copy.deepcopy(request))
			# logging.info(resp)
			requester_total += 1
			# current_concurrent -= 1
			# total_genned += 1
			# logging.info("RESPONSE: ", resp)
			# yield resp

	async def make_request(request):
		logging.info("REQUEST: ", request)
		while True:
			thing = await server.requester(request)
			if type(thing) == int:
				await asyncio.sleep(1)
				continue
			break
		logging.info(thing)
		return thing

	async def test_all_requesting():
		coros = [make_request(copy.deepcopy(test_request), ) for _ in range(n_per_batch)]
		results = await asyncio.gather(*coros)
		return results
	
	def make_request(request):
		logging.info("REQUEST: ", request)
		while True:
			thing = requests.get(f'http://{server_config["server_host"]}:{server_config["server_port"]}/request', json = request).json()
			if type(thing) == int:
				continue
			break
		logging.info(thing)
		return thing
	# time.sleep(10000000)
	logging.info("STARTING BENCHMARK")
	tstart = time.perf_counter()
	# for i in range(int(n_to_run / n_per_batch)):
	# 	asyncio.run(test_all_requesting())
	# asyncio.run(cont_request(test_request))
	# asyncio.run(cont_request_manager(test_request))
	make_request(test_request)
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

# def start_server(engine_configs, server_config):
# 	server = Server (
# 		engine_configs = engine_configs,
# 		server_config = server_config,
# 	)
# 	server.start()

class JetThread(threading.Thread):
	"""Thread that kills the program if it fails.

	If a driver thread goes down, we can't operate.
	"""

	def run(self):
		try:
			super().run()
		except Exception as e:  # pylint: disable=broad-exception-caught
			logging.info(f'Thread {self.name} encountered an error: {e}')
			traceback.print_exc()
			os.kill(os.getpid(), signal.SIGKILL)

class Server:
	def __init__(self, engine_configs: list, server_config: dict):
		'''
		Initialises the BaseServer class.
		Takes in the MaxEngine config as engine_config.
		Takes in a dictionary of server settings as server_config.
		'''
		
		self.engine_configs = engine_configs
		self._parse_and_validate_config(server_config)

		if self.DEBUG:
			BaseServer = logging.getLogger()
			BaseServer.setLevel(logging.INFO)
			logging.info('loaded engine and server configs')
		
		# self.start()

		# if self.DEBUG:
		# 	logging.info('initialised server')

	def start(self):
		'''
		Starts the BaseServer.
		'''
		initialise_tracking()
		self.live = False
		self.accept_requests = True

		if self.DEBUG:
			root = logging.getLogger()
			root.setLevel(logging.INFO)
			logging.info('started tracking memory usage')
			ts = time.perf_counter()
			t_start_start = ts

		self.engines = [maxengine.MaxEngine(engine_config) for engine_config in self.engine_configs]
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self.engines)} engines in {te-ts:4f} seconds')
			ts = te

		self.params = [engine.load_params() for engine in self.engines]
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self.params)} params in {te-ts:4f} seconds')
			ts = te
		
		self.decode_states = [engine.init_decode_state() for engine in self.engines]
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'initialised {len(self.decode_states)} decode states in {te-ts:4f} seconds')
			ts = te

		self._load_tokenizer()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded tokenizer in {te-ts:4f} seconds')
			ts = te
		
		self.batch_sizes = [engine.max_concurrent_decodes for engine in self.engines]

		# this does not need a lock as it is only in a single thread all the time.
		# however, to support functionality of blocking prefill when generating
		# we still use the lock. The lock has negligible overhead compared to generation.
		self._decode_state_locks = [threading.Lock() for _ in range(self.n_generate_threads)]

		self._request_queue = mpqueue(maxsize = math.ceil(self.request_max_store / 2))
		self._untokenized_request_queue = mpqueue(maxsize = math.ceil(self.request_max_store / 2))

		self._slots = [[copy.deepcopy(default_slot) for _ in range(batch_size)] for batch_size in self.batch_sizes]
		self._slots_locks = [threading.Lock() for _ in range(self.n_generate_threads)]
		self._slots_freed_events = [threading.Event() for _ in range(self.n_generate_threads)]

		self._sampled_tokens_queues = [queue.Queue(maxsize = self.sampled_tokens_max_store) for _ in range(self.n_generate_threads)]

		self._response_queue = mpqueue(maxsize = self.response_max_store)
		self._detokenized_response_queues = {idx: mpqueue(maxsize = 1) for idx in range(self.n_request_ids)}

		self.current_concurrent_requests = 0
		self.available_request_ids = [idx for idx in range(self.n_request_ids)]

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded all synchronisers in {te-ts:4f} seconds')
			ts = te

		# start threads
		self.live = True

		self._generate_threads = [
			JetThread(
				target = self._generate_thread,
				name = f'generate_thread_{i}',
				args = (i, ),
			) for i in range(self.n_generate_threads)
		]

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self._generate_threads)} generate threads in {te-ts:4f} seconds')
			ts = te

		self._processing_threads = [
			JetThread(
				target = self._processing_thread,
				name = f'processing_thread_{i}',
				args = (i, ),
			) for i in range(self.n_generate_threads)
		]

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self._processing_threads)} processing threads in {te-ts:4f} seconds')
			ts = te
		
		self._place_request_processes = [
			mpprocess(
				target = self._place_request_process,
			) for i in range(self.n_tokenization_processes)
		]

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self._place_request_processes)} place request processes in {te-ts:4f} seconds')
			ts = te
		
		self._response_parser_processes = [
			mpprocess(
				target = self._response_parser_process,
				args = (i, ),
			) for i in range(self.n_detokenization_processes)
		]

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded {len(self._response_parser_processes)} response parser processes in {te-ts:4f} seconds')
			ts = te
		
		for thread in self._generate_threads:
			thread.daemon = True
			thread.start()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'started {len(self._generate_threads)} generate threads in {te-ts:4f} seconds')
			ts = te

		for thread in self._processing_threads:
			thread.daemon = True
			thread.start()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'started {len(self._processing_threads)} processing threads in {te-ts:4f} seconds')
			ts = te
		
		for process in self._place_request_processes:
			process.daemon = True
			process.start()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'started {len(self._place_request_processes)} place request processes in {te-ts:4f} seconds')
			ts = te
		
		for process in self._response_parser_processes:
			process.daemon = True
			process.start()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'started {len(self._response_parser_processes)} response parser processes in {te-ts:4f} seconds')
			ts = te

		# kick off server
		self.accept_requests = True

		# self.app = FastAPI()
		# self.app.add_route('/request', self.requester)
		# uvicorn.run(self.app, host = self.server_host, port = self.server_port)
		# # self.app = web.Application()
		# # self.app.add_routes([web.get('/request', self.requester)])
		# # self.app.add_routes([web.get('/shutdown', self.shutdown)])
		# # web.run_app(self.app, host = self.server_host, port = self.server_port)
		# if self.DEBUG:
		# 	logging.info(f'started server in {time.perf_counter()-t_start_start:4f} seconds')

	def stop(self):
		'''
		Stops the server gracefully.
		'''
		self.accept_requests = False
		while True:
			# wait until requests are all cleared
			if not self._request_queue.empty():
				continue
			# wait until insertion backlog is all cleared
			for idx in range(self.n_generate_threads):
				for idx2 in range(self.n_prefill_threads_per_engine):
					for queue in self._prefill_queues[idx][idx2]:
						if not queue.empty():
							continue
			# wait until generation backlog is all cleared
			for idx, lock in enumerate(self._slots_locks):
				with lock:
					for slot in self._slots[idx]:
						if slot['active']:
							continue
			# wait until processing backlog is all cleared
			for idx in range(self.n_generate_threads):
				if not self._sampled_tokens_queues[idx].empty():
					continue
			# wait until all responses are returned
			if not self._response_queue.empty():
				continue
			break
		self.live = False
		try:
			requests.get(f'http://{self.server_host}:{self.server_port}/shutdown')
		except:
			pass

	async def shutdown(self):
		raise web.GracefulExit()
	
	def _generate_thread(self, idx):
		'''
		Generate thread for an engine.
		Workflow:
		1. check for free slots
		2. check for ready prefills
		3. insert any ready prefills into any free slots
		4. check for ready generation steps (my_local_waiting_queue full)
		5. puts any ready generation steps into processing queue
		6. generates
		7. copies results to host async, puts results into waiting queue
		8. repeat as long as self.live
		'''
		def fix_numpy(arr):
			nparr = np.array(arr)
			return np.where(nparr[..., 1] == 1, nparr[..., 0], 0)
		my_generation_steps = 0
		sampled_tokens_prev = None
		if self.DEBUG:
			logging.info(f'engine {idx} ready')
			ts = time.perf_counter()
			t_of_last_loop = ts
			t_total = 0
			t_gen_total = 0
			t_start_thread = ts
		while self.live:
			# check if there are free slots we can insert into
			successfully_inserted = 0
			t_of_last_get = time.perf_counter()
			while self._slots_freed_events[idx].is_set() and successfully_inserted < self.max_one_time_insertion:
				# and check if there are any tokenized ready to prefill
				can_insert = False
				try:
					input_ids, attention_mask, true_length, token_positions, formatted_request = self._request_queue.get(block = False)
					input_ids = jnp.array(input_ids, dtype = jnp.int32)
					attention_mask = jnp.array(attention_mask, dtype = jnp.int32)
					token_positions = jnp.array(token_positions, dtype = jnp.int32)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} got request in {te-ts:4f} seconds')
						ts = te
					can_insert = True
					t_of_last_get = time.perf_counter()
				except queue.Empty:
					if time.perf_counter() - t_of_last_get > self.insertion_timeout:
						if self.DEBUG:
							te = time.perf_counter()
							logging.info(f'engine {idx} timed out waiting for tokenized request in {te-ts:4f} seconds')
							ts = te
						break
					continue
				if can_insert:
					# find slot to insert into
					inserted = False
					free_slot_left = False
					with self._slots_locks[idx]:
						for slot_idx, slot in enumerate(self._slots[idx]):
							if slot['active']:
								# if self.DEBUG:
								# 	te = time.perf_counter()
								# 	logging.info(f'engine {idx} slot {slot_idx} is active')
								# 	ts = te
								continue
							# insert. we do not need to lock decode state as prefill does not
							# alter the decode state and the lock is only used to prevent
							# prefilling while generating based on config

							prefill_result = self.engines[idx].prefill(
								params = self.params[idx],
								padded_tokens = input_ids,
								attention_mask = attention_mask,
								token_positions = token_positions,
								true_length = true_length,
							)

							if self.DEBUG:
								te = time.perf_counter()
								logging.info(f'engine {idx} prefilled into slot {slot_idx} in {te-ts:4f} seconds')
								ts = te

							self.decode_states[idx] = self.engines[idx].insert (
								prefill_result,
								self.decode_states[idx],
								slot = slot_idx,
							)
							inserted = True
							successfully_inserted += 1
							if self.DEBUG:
								te = time.perf_counter()
								logging.info(f'engine {idx} inserted into slot {slot_idx} in {te-ts:4f} seconds')
								ts = te
							
							self._slots[idx][slot_idx] = formatted_request
							self._slots[idx][slot_idx]['request_timestep'] = my_generation_steps

							if self.DEBUG:
								te = time.perf_counter()
								logging.info(f'engine {idx} updated slot {slot_idx} in {te-ts:4f} seconds')
								ts = te

							free_slot_left = False
							for slot in self._slots[idx]:
								if not slot['active']:
									free_slot_left = True

							break
						if not inserted:
							# something went horribly wrong
							raise Exception(f"generate thread {idx} failed to insert prefill, exiting")
						if not free_slot_left:
							# no more free slots, clear event
							self._slots_freed_events[idx].clear()
				t_of_last_get = time.perf_counter()
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'engine {idx} inserted {successfully_inserted} prefills in {te-ts:4f} seconds')
				ts = te
			
			# lock to prevent prefill and generation from occuring simultaneously if configured
			with self._decode_state_locks[idx]:
				if self.DEBUG:
					te = time.perf_counter()
					logging.info(f'engine {idx} locked decode state in {te-ts:4f} seconds')
					ts = te
					t_start_gen = te
				self.decode_states[idx], sampled_tokens_new = self.engines[idx].generate(
					self.params[idx],
					self.decode_states[idx],
					# sampling_strategy = self.sampling_strategy,
					# topk = self.top_k,
					# nucleus_topp = self.top_p,
					# temperature = self.temperature,
				)
				if self.DEBUG:
					te = time.perf_counter()
					logging.info(f'engine {idx} only generation in {te-ts:4f} seconds')
					ts = te
				if sampled_tokens_prev:
					sampled_tokens_prev = sampled_tokens_prev.data.block_until_ready()
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens block until ready in {te-ts:4f} seconds')
						ts = te
					# add_to_slots(my_generation_steps - 1, fix(sampled_tokens_prev))
					# logging.info(sampled_tokens_prev)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens printed in {te-ts:4f} seconds')
						ts = te
					# fixed = fix(sampled_tokens_prev)
					fixed = fix_numpy(sampled_tokens_prev)
					# logging.info(fixed)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens fixed in {te-ts:4f} seconds')
						ts = te
					# fixed = logging.info(fixed)
					# if self.DEBUG:
					# 	te = time.perf_counter()
					# 	logging.info(f'engine {idx} sampled tokens block until ready in {te-ts:4f} seconds')
					# 	ts = te
					self._sampled_tokens_queues[idx].put((my_generation_steps - 1, fixed))
					# add_to_slots(my_generation_steps - 1, fixed)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens put to sampled tokens queue in {te-ts:4f} seconds')
						ts = te
				sampled_tokens_new.copy_to_host_async()
				if self.DEBUG:
					te = time.perf_counter()
					logging.info(f'engine {idx} sampled tokens copy to host in {te-ts:4f} seconds')
					ts = te
				sampled_tokens_prev = sampled_tokens_new
			# "read" the request after a delay of n_generation_cycles_wait cycles
			# to eliminate extra overhead from generate function and copying over to host
			# if my_local_waiting_queue.full():
			# 	# timestep, sampled_tokens_to_put = my_local_waiting_queue.get()
			# 	# sampled_tokens_to_put.data.block_until_ready()
			# 	# if self.DEBUG:
			# 	# 	te = time.perf_counter()
			# 	# 	logging.info(f'engine {idx} blocked sampled tokens until ready to sampled tokens queue in {te-ts:4f} seconds')
			# 	# 	ts = te
			# 	# self._sampled_tokens_queues[idx].put((timestep, sampled_tokens_to_put))
			# 	self._sampled_tokens_queues[idx].put(my_local_waiting_queue.get())
			# 	if self.DEBUG:
			# 		te = time.perf_counter()
			# 		logging.info(f'engine {idx} put sampled tokens to sampled tokens queue in {te-ts:4f} seconds')
			# 		ts = te
			# copy to host asynchronously while waiting to make processing very fast
			# sampled_tokens.copy_to_host_async()
			# my_local_waiting_queue.put((my_generation_steps, sampled_tokens))
			my_generation_steps += 1
			if self.DEBUG:
				te = time.perf_counter()
				t_total += te-t_of_last_loop
				t_gen_total += te-t_start_gen
				t_absolute_total = te - t_start_thread
				logging.info(f'engine {idx} generation steps: {my_generation_steps}')
				logging.info(f'engine {idx} total generation cycle time: {te-t_of_last_loop:4f}')
				logging.info(f'engine {idx} generation cycles per second: {1/(te-t_of_last_loop):4f}')
				logging.info(f'engine {idx} total generation time: {t_total:4f}')
				logging.info(f'engine {idx} total generation cycles per second: {my_generation_steps/t_total:4f}')
				logging.info(f'engine {idx} total generation time without insertion: {t_gen_total:4f}')
				logging.info(f'engine {idx} total generation cycles per second without insertion: {my_generation_steps/t_gen_total:4f}')
				logging.info(f'engine {idx} total absolute time: {t_absolute_total:4f}')
				logging.info(f'engine {idx} total absolute cycles per second: {my_generation_steps/t_absolute_total:4f}')
				ts = te
				t_of_last_loop = te
		logging.info(f"generate thread {idx} exiting")

	def _processing_thread(self, idx):
		def add_to_slots(timestep, arr):
			n_free_slots = 0
			with self._slots_locks[idx]:
				for i, tok in enumerate(arr):
					if not self._slots[idx][i]['active']:
						n_free_slots += 1
						# if self.DEBUG:
						# 	logging.info(f"slot {i} is not active")
						continue
					tok = int(tok)
					if self._slots[idx][i]['request_timestep'] == timestep:
						self._slots[idx][i]['output_tokens'] == [tok]
					elif self._slots[idx][i]['request_timestep'] < timestep:
						self._slots[idx][i]['output_tokens'].append(tok)
					
					stop = False
					if self._slots[idx][i]['request_config']['stop_on_eos'] and tok == self._slots[idx][i]['request_config']['eos_token_id']:
						stop = True
						end_reason = 'eos'
					elif self._slots[idx][i]['request_config']['max_tokens'] is not None and len(self._slots[idx][i]['output_tokens']) + len(self._slots[idx][i]['input_tokens']) >= self._slots[idx][i]['request_config']['max_tokens']:
						stop = True
						end_reason = 'max_tokens'
					elif self._slots[idx][i]['request_config']['max_new_tokens'] is not None and len(self._slots[idx][i]['output_tokens']) >= self._slots[idx][i]['request_config']['max_new_tokens']:
						stop = True
						end_reason = 'max_new_tokens'
					if stop:
						# detokenize, format into response and put onto queue
						response = {
							'request_config': copy.deepcopy(self._slots[idx][i]['request_config']),
							'end_reason': end_reason,
							'request_timestep': copy.deepcopy(self._slots[idx][i]['request_timestep']),
							'request_id': copy.deepcopy(self._slots[idx][i]['request_id']),
							'input_tokens': copy.deepcopy(self._slots[idx][i]['input_tokens']),
							'input_sequence': copy.deepcopy(self._slots[idx][i]['input_sequence']),
							'output_tokens': copy.deepcopy(self._slots[idx][i]['output_tokens']),
							'output_sequence': '',
							# 'output_sequence': self.detokenize(copy.deepcopy(self._slots[idx][i]['output_tokens']), self._slots[idx][i]['request_config']['clean_up_tokenization_spaces'], self._slots[idx][i]['request_config']['skip_special_tokens']),
						}

						self._slots[idx][i] = copy.deepcopy(default_slot)
						self._slots[idx][i]['active'] = False
						
						self._response_queue.put(response)
						
						n_free_slots += 1
						if self.DEBUG:
							logging.info(f'processing {idx} slot {i} finished due to: {end_reason}')
							# logging.info(response)
			return n_free_slots
		if self.DEBUG:
			ts = time.perf_counter()
			t_of_last_loop = ts
			logging.info(f'processing {idx} ready')
		while True:
			timestep, sampled_tokens = self._sampled_tokens_queues[idx].get()
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'processing {idx} got sampled tokens in {te-ts:4f} seconds')
				ts = te
			n_free_slots = add_to_slots(timestep, sampled_tokens)
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'processing {idx} added to slots in {te-ts:4f} seconds')
				logging.info(f'processing {idx} total processing cycle time: {te-t_of_last_loop:4f}')
				logging.info(f'processing {idx} n free slots: {n_free_slots}')
				ts = te
				t_of_last_loop = te
			
			# get the number of prefill threads allowed to insert
			if n_free_slots > 0:
				self._slots_freed_events[idx].set()
			else:
				self._slots_freed_events[idx].clear()
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'processing {idx} set slot free event in {te-ts:4f} seconds')
				ts = te

	def _place_request_process(self):
		while self.live:
			if not self.accept_requests:
				if self.DEBUG:
					logging.info("server not accepting requests")
				continue
			if self.DEBUG:
				logging.info("server accepted request")
			request = self._untokenized_request_queue.get()
			request = self.tokenize(request)
			input_ids = request['input_ids']
			# input_ids = np.asarray(input_ids, dtype = np.int32)
			attention_mask = request['attention_mask']
			# attention_mask = np.asarray(attention_mask, dtype = np.int32)
			true_length = request['true_length']
			token_positions = request['token_positions']
			# token_positions = np.asarray(token_positions, dtype = np.int32)
			formatted_request  = request['formatted_request']
			self._request_queue.put((input_ids, attention_mask, true_length, token_positions, formatted_request))
			if self.DEBUG:
				logging.info("server placed request on queue")

	async def requester(self, request):
		# get request id
		if not self.live:
			if self.DEBUG:
				logging.info(f'server not live')
			return 0
		if self.current_concurrent_requests >= self.max_concurrent_requests:
			# if self.DEBUG:
			# 	logging.info(f'connection throttled')
			return 1
		if len(self.available_request_ids) == 0:
			# if self.DEBUG:
			# 	logging.info(f'no available request ids')
			return 2
		self.current_concurrent_requests += 1
		idx = self.available_request_ids.pop(0)
		request = json.loads(request)
		request['request_id'] = idx
		if self.DEBUG:
			ts = time.perf_counter()
			logging.info(f"request {idx} getting tokenized")

		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f"request {idx} tokenized in {te-ts:4f} seconds")
			ts = te

		# send request to base server
		self._untokenized_request_queue.put(request)
		
		if self.DEBUG:
			logging.info(f"request {idx} sent to base server")

		# wait for response
		while True:
			try:
				response = self._detokenized_response_queues[idx].get(block = False)
				if self.DEBUG:
					te = time.perf_counter()
					logging.info(f"request {idx} got response in {te-ts:4f} seconds")
					ts = te
				break
			except:
				await asyncio.sleep(1)
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f"request {idx} got response in {te-ts:4f} seconds")
			ts = te

		# cleanup
		self.available_request_ids.append(idx)
		self.current_concurrent_requests -= 1

		return response
		# return web.json_response(response)
	
	async def fastapi_requester(self, request):
		res = await self.requester(request)
		if res == 0:
			asyncio.sleep(self.delay_before_reject)
			raise HTTPException(503, "Server not live.")
		if res == 1:
			asyncio.sleep(self.delay_before_reject)
			raise HTTPException(429, "Exceeded max concurrent requests.")
		if res == 2:
			asyncio.sleep(self.delay_before_reject)
			raise HTTPException(429, "Exceeded maximum request ids.")
		return res
	
	def start_fastapi_server(self, host = None, port = None):
		if host is None:
			host = self.server_host
		if port is None:
			port = self.server_port
		if self.DEBUG:
			logging.info(f"starting fastapi server on {host}:{port}")
		self.fastapi_app = FastAPI(debug = self.DEBUG)
		self.fastapi_app.add_api_route("/request", self.fastapi_requester, methods = ["GET"])
		uvicorn.run(self.fastapi_app, host = host, port = port)
	
	def _response_parser_process(self, idx):
		while self.live:
			# time.sleep(10)
			response = self._response_queue.get()

			if self.DEBUG:
				ts = time.perf_counter()
				logging.info(f"response parser {idx} response {response['request_id']} getting detokenized")
			out_sequence = self.detokenize (
				copy.deepcopy(response['output_tokens']),
				response['request_config']['clean_up_tokenization_spaces'],
				response['request_config']['skip_special_tokens']
			)
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f"response parser {idx} response {response['request_id']} got detokenized in {te-ts:4f} seconds")
				ts = te
			response['output_sequence'] = out_sequence
			self._detokenized_response_queues[response['request_id']].put(response)
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f"response parser {idx} response {response['request_id']} put in response queue in {te-ts:4f} seconds")
				ts = te
	
	def _load_tokenizer(self):
		tokenizer = transformers.AutoTokenizer.from_pretrained (
			self.tokenizer_config['path'],
			use_fast = self.tokenizer_config['use_fast'],
		)
		tokenizer.pad_token_id = self.tokenizer_config['pad_token_id']
		tokenizer.bos_token_id = self.tokenizer_config['bos_token_id']
		tokenizer.eos_token_id = self.tokenizer_config['eos_token_id']
		tokenizer.padding_side = self.tokenizer_config['padding_side']
		self.tokenizer = tokenizer
		self.tokenizer_config['possible_lengths'] = self.tokenizer_config['possible_lengths'][:self.tokenizer_config['possible_lengths'].index(self.max_prefill_length) + 1]

	def tokenize(self, request: dict):
		request_max_length = self.max_prefill_length
		for val in self.tokenizer_config['possible_lengths']:
			if val >= request['config']['max_tokens']:
				request_max_length = val
				break
		tokenized = self.tokenizer (
			request['input_sequence'],
			padding = True,
			truncation = True,
			max_length = min(request_max_length, self.max_sequence_length),
			pad_to_multiple_of = min(request_max_length, self.max_sequence_length),
			return_tensors = "np",
			)
		input_ids = tokenized.input_ids[0]
		attention_mask = tokenized.attention_mask[0]
		true_length = np.count_nonzero(attention_mask)
		nearest_length = take_nearest_length (
			self.tokenizer_config['possible_lengths'][:self.tokenizer_config['possible_lengths'].index(request_max_length) + 1],
			true_length,
		)
		input_ids = input_ids[:nearest_length]
		attention_mask = attention_mask[:nearest_length]
		true_length = np.count_nonzero(attention_mask)
		token_positions = np.arange(0, input_ids.shape[0])
		formatted_request = {
			'active': True,
			'request_id': request['request_id'], # this is for use for the server
			'request_timestep': -1, # the timestep where the request was inserted. this is for use for the processor (-1 means to ignore)
			'request_config': {
				'stop_on_eos': request['config']['stop_on_eos'],
				'eos_token_id': request['config']['eos'],
				'max_tokens': request['config']['max_tokens'],
				'max_input_tokens': request['config']['max_input_tokens'],
				'max_new_tokens': request['config']['max_new_tokens'],
				'clean_up_tokenization_spaces': request['config']['clean_up_tokenization_spaces'],
				'skip_special_tokens': request['config']['skip_special_tokens'],

				# individual sampling is unused due to bad performance, all requests will follow the server_config sampling
				'sampling_strategy': request['config']['sampling_strategy'],
				'top_k': request['config']['top_k'],
				'top_p': request['config']['top_p'],
				'temperature': request['config']['temperature'],
			},
			'input_tokens': input_ids[:true_length].tolist() if self.tokenizer_config['padding_side'] == 'right' else input_ids[-true_length:].tolist(),
			'input_sequence': request['input_sequence'],
			'output_tokens': [],
			'output_sequence': '', # unused for now
		}
		full_formatted_request = {
			'input_ids': input_ids.tolist(),
			'attention_mask': attention_mask.tolist(),
			'true_length': true_length,
			'token_positions': token_positions.tolist(),
			'formatted_request': formatted_request,
		}
		return full_formatted_request
	
	def detokenize(self, input_ids, clean_up_tokenization_spaces, skip_special_tokens):
		return self.tokenizer.decode (
			input_ids,
			clean_up_tokenization_spaces = clean_up_tokenization_spaces,
			skip_special_tokens = skip_special_tokens,
			)

	def _parse_and_validate_config(self, server_config):
		self.n_generation_cycles_wait = server_config['n_generation_cycles_wait']
		assert self.n_generation_cycles_wait >= 1, "server config n_generation_cycles_wait must be >= 1"
		if self.n_generation_cycles_wait < 32:
			warnings.warn("""########################SERVER WARNING########################
SERVER CONFIG 'n_generation_cycles_wait' IS LESS THAN 128.
PERFORMANCE COULD BE LOWER THAN EXPECTED.""")

		self.DEBUG = server_config['DEBUG']
		assert self.DEBUG in [True, False], "server config DEBUG must be True or False"

		self.stop_on_eos = server_config['stop_on_eos']
		assert self.stop_on_eos in [True, False], "server config stop_on_eos must be True or False"

		self.n_generate_threads = server_config['n_generate_threads']
		assert self.n_generate_threads >= 1, "server config n_generate_threads must be >= 1"
		assert self.n_generate_threads == len(self.engine_configs), "server config n_generate_threads must be equal to the number of engine_configs"

		self.n_tokenization_processes = server_config['n_tokenize_processes']
		assert self.n_tokenization_processes >= 1, "server config n_tokenize_processes must be >= 1"

		self.n_detokenization_processes = server_config['n_detokenize_processes']
		assert self.n_detokenization_processes >= 1, "server config n_detokenize_processes must be >= 1"

		self.prefill_while_generating = server_config['prefill_while_generating']
		assert self.prefill_while_generating in [True, False], "server config prefill_while_generating must be True or False"

		self.n_prefill_threads_per_engine = server_config['n_prefill_threads_per_engine']
		assert self.n_prefill_threads_per_engine >= 1, "server config n_prefill_threads_per_engine must be >= 1"

		self.prefill_batch_size = server_config['prefill_batch_size']
		assert self.prefill_batch_size == 1, "batched prefill not implemented yet, must be 1"

		self.prefill_max_store = server_config['prefill_max_store']
		assert self.prefill_max_store >= self.prefill_batch_size, "server config prefill_max_store must be >= prefill_batch_size"

		self.prefill_request_get_timeout = server_config['prefill_request_get_timeout']
		assert self.prefill_request_get_timeout > 0, "server config prefill_request_get_timeout must be > 0"
		if self.prefill_request_get_timeout > 0.001:
			warnings.warn("""########################SERVER WARNING########################
SERVER CONFIG 'prefill_request_get_timeout' IS MORE THAN 0.001.
THIS DIRECTLY ADDS TO EVERY GENERATION LOOP.""")

		self.max_one_time_insertion = server_config['max_one_time_insertion']
		assert self.max_one_time_insertion >= 1, "server config max_one_time_insertion must be >= 1"

		self.insertion_timeout = server_config['insertion_timeout']
		assert self.insertion_timeout > 0, "server config insertion_timeout must be > 0"

		self.request_max_store = server_config['request_max_store']
		assert self.request_max_store >= 1, "server config request_max_store must be >= 1"

		self.response_max_store = server_config['response_max_store']
		assert self.response_max_store >= 1, "server config response_max_store must be >= 1"

		self.sampled_tokens_max_store = server_config['sampled_tokens_max_store']
		assert self.sampled_tokens_max_store >= 1, "server config sampled_tokens_max_store must be >= 1"

		self.max_sequence_length = server_config['max_sequence_length']
		assert self.max_sequence_length >= 2, "server config max_sequence_length must be >= 2"

		self.max_prefill_length = server_config['max_prefill_length']
		assert self.max_prefill_length >= 1, "server config max_prefill_length must be >= 1"

		self.sampling_strategy = server_config['sampling_strategy']
		assert self.sampling_strategy in ["greedy", "weighted", "topk", "nucleus"], "server config sampling_strategy must be 'greedy', 'weighted', 'topk', or 'nucleus'"
		
		self.top_k = server_config['top_k']
		if self.sampling_strategy == "top_k":
			assert self.top_k >= 1, "server config top_k must be >= 1"

		self.top_p = server_config['nucleus_top_p']
		if self.sampling_strategy == "nucleus":
			assert self.top_p > 0, "server config nucleus_top_p must be between 0 and 1"
			assert self.top_p <= 1, "server config nucleus_top_p must be between 0 and 1"

		self.temperature = server_config['temperature']
		if not self.sampling_strategy == "greedy":
			assert self.temperature > 0, "server config temperature must be > 0"

		self.tokenizer_config = server_config['tokenizer_config']
		assert self.tokenizer_config['use_fast'] in [True, False], "server config tokenizer_config use_fast must be True or False"
		assert self.tokenizer_config['padding_side'] in ["left", "right"], "server config tokenizer_config padding_side must be 'left' or 'right'"

		self.server_host = server_config['server_host']
		self.server_port = server_config['server_port']
		self.n_request_ids = server_config['n_request_ids']
		self.max_concurrent_requests = server_config['max_concurrent_requests']
		self.delay_before_reject = server_config['delay_before_reject']

def validate_config(config):
	assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
																						"Using generate_param_only_checkpoint."

if __name__ == "__main__":
	jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
	pyconfig.initialize(sys.argv)
	cfg = pyconfig.config
	validate_config(cfg)
	main(engine_configs=[cfg], server_config=server_cfg)