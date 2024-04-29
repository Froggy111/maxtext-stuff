'''
Implementation of continous batching with MaxEngine.
Code is based on JetStream's orchestrator.
Currently supports:
Async API (Server.request())
'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "maxtext/MaxText"))

import jax, transformers, time, random, warnings,\
	threading, queue, copy, traceback, signal, logging, asyncio

import numpy as np
from jax import numpy as jnp

from jax_smi import initialise_tracking

import maxengine

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
	'n_tokenize_threads': 1, # total number of tokenizing threads. 1 should be usually enough.
	# IGNORE THE PREFILL THREAD SETTINGS. THEY ARE UNUSED
	'n_prefill_threads_per_engine': 1, # number of prefill threads per engine
	'prefill_batch_size': 1, # batch size when prefilling, NOT IMPLEMENTED YET, DO NOT MODIFY
	'prefill_max_store': 1, # how many prefill results to store per thread (at least 1)
	'prefill_request_get_timeout': 0.1, # timeout for getting a request from request queue to prefill in seconds

	'max_one_time_insertion': 384, # max number of insertions per generate cycle
	'insertion_timeout': 0.1, # timeout for recieving a next prefill result when inserting, in seconds
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

	# tokenizer settings
	'tokenizer_config': {
		'path': "/home/ljy/tokenizer",
		'use_fast': True,
		'padding_side': "right",
		'pad_token_id': 2,
		'bos_token_id': 1,
		'eos_token_id': 3,
		'unk_token_id': 0,
		'possible_lengths': [#16, 32, 64, 128, 256, 512,
					   1024, 2048, 4096, 8192, 16384, 32768,],
	},
}

def main(engine_configs, server_config):
	server = Server (
		engine_configs = engine_configs,
		server_config = server_config,
	)
	test_request = copy.deepcopy(request_template)
	n_to_run = 100000
	n_per_batch = 1000
	n_prefill_toks = 1024
	n_generate_toks = 1024
	async def test_requesting(request, idx):
		while True:
			request['request_id'] = idx
			resp = await server.request(request, 0.1)
			if not resp:
				await asyncio.sleep(1)
				continue
			# print(resp)
			break
	async def test_all_requesting():
		coros = [test_requesting(copy.deepcopy(test_request), idx) for idx in range(n_per_batch)]
		results = await asyncio.gather(*coros)
		return results
	tstart = time.perf_counter()
	for i in range(int(n_to_run / n_per_batch)):
		asyncio.run(test_all_requesting())
	tend = time.perf_counter()
	queries_per_second = n_to_run / (tend - tstart)
	prefill_toks_per_second = n_prefill_toks * n_to_run / (tend - tstart)
	generate_toks_per_second = n_generate_toks * n_to_run / (tend - tstart)
	print(f'Queries per second: {queries_per_second}')
	print(f'Prefill tokens per second: {prefill_toks_per_second}')
	print(f'Generate tokens per second: {generate_toks_per_second}')
	exit()

class JetThread(threading.Thread):
  """Thread that kills the program if it fails.

  If a driver thread goes down, we can't operate.
  """

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Thread {self.name} encountered an error: {e}')
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)

class Server:
	def __init__(self, engine_configs: list, server_config: dict):
		'''
		Initialises the server.
		Takes in the MaxEngine config as engine_config.
		Takes in a dictionary of server settings as server_config.
		'''
		
		self.engine_configs = engine_configs
		self._parse_and_validate_config(server_config)
		initialise_tracking()

		if self.DEBUG:
			root = logging.getLogger()
			root.setLevel(logging.INFO)
			logging.info('loaded engine and server configs, started tracking memory usage')
			ts = time.perf_counter()
			t_start_init = ts

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
		
		self.batch_sizes = [engine.max_concurrent_decodes for engine in self.engines]
		self._load_tokenizer()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'loaded tokenizer in {te-ts:4f} seconds')
			ts = te

		# this does not need a lock as it is only in a single thread all the time.
		# however, to support functionality of blocking prefill when generating
		# we still use the lock. The lock has negligible overhead compared to generation.
		self._decode_state_locks = [threading.Lock() for _ in range(self.n_generate_threads)]

		self._request_queue = queue.Queue(maxsize = self.request_max_store / 2)
		self._tokenized_request_queue = queue.Queue(maxsize = self.request_max_store / 2)

		# self._prefill_queues = [[queue.Queue(maxsize = self.prefill_max_store) for _ in range(self.n_prefill_threads_per_engine)] for _ in range(self.n_generate_threads)]

		self._slots = [[copy.deepcopy(default_slot) for _ in range(batch_size)] for batch_size in self.batch_sizes]
		self._slots_locks = [threading.Lock() for _ in range(self.n_generate_threads)]
		self._slots_freed_events = [threading.Event() for _ in range(self.n_generate_threads)]

		self._sampled_tokens_queues = [queue.Queue(maxsize = self.sampled_tokens_max_store) for _ in range(self.n_generate_threads)]

		self._response_queue = queue.Queue(maxsize = self.response_max_store)

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
			logging.info(f'loaded {len(self._processing_threads)} processing threads in {te-ts:4f} seconds')
			ts = te
		
		self._tokenize_threads = [
			JetThread(
				target = self._tokenize_thread,
				name = f'tokenize_thread_{i}',
				args = (i, ),
			) for i in range(self.n_tokenize_threads)
		]

		# self._prefill_threads = [[
		# 	JetThread(
		# 		target = self._prefill_thread,
		# 		name = f'prefill_thread_{i2} in engine {i}',
		# 		args = (i, i2),
		# 	) for i2 in range(self.n_prefill_threads_per_engine)
		# ] for i in range(self.n_generate_threads)]

		# if self.DEBUG:
		# 	te = time.perf_counter()
		# 	logging.info(f'loaded {len(self._prefill_threads)} prefill threads in {te-ts:4f} seconds')
		# 	ts = te

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

		for thread in self._tokenize_threads:
			thread.daemon = True
			thread.start()
		if self.DEBUG:
			te = time.perf_counter()
			logging.info(f'started {len(self._tokenize_threads)} tokenize threads in {te-ts:4f} seconds')
			ts = te
		
		# for threadlist in self._prefill_threads:
		# 	for thread in threadlist:
		# 		thread.daemon = True
		# 		thread.start()
		# if self.DEBUG:
		# 	te = time.perf_counter()
		# 	logging.info(f'started {len(self._prefill_threads)} prefill threads in {te-ts:4f} seconds')
		# 	ts = te
		
		# kick off server
		self.accept_requests = True
		if self.DEBUG:
			logging.info(f'initialised server in {time.perf_counter()-t_start_init:4f} seconds')

	def stop(self):
		'''
		Stops the server gracefully.
		'''
		self.accept_requests = False
		while True:
			# wait until requests are all cleared
			if not self._request_queue.empty():
				continue
			if not self._tokenized_request_queue.empty():
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
		while self.live:
			# check if there are free slots we can insert into
			successfully_inserted = 0
			t_of_last_get = time.perf_counter()
			while self._slots_freed_events[idx].is_set() and successfully_inserted < self.max_one_time_insertion:
				# and check if there are any tokenized ready to prefill
				can_insert = False
				try:
					input_ids, attention_mask, true_length, token_positions, formatted_request = self._tokenized_request_queue.get_nowait()
					can_insert = True
					t_of_last_get = time.perf_counter()
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} got tokenized request in {te-ts:4f} seconds')
						ts = te
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
								if self.DEBUG:
									te = time.perf_counter()
									logging.info(f'engine {idx} slot {slot_idx} is active')
									ts = te
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
				self.decode_states[idx], sampled_tokens_new = self.engines[idx].generate(
					self.params[idx],
					self.decode_states[idx],
					sampling_strategy = self.sampling_strategy,
					topk = self.top_k,
					nucleus_topp = self.top_p,
					temperature = self.temperature,
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
					# print(sampled_tokens_prev)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens printed in {te-ts:4f} seconds')
						ts = te
					# fixed = fix(sampled_tokens_prev)
					fixed = fix_numpy(sampled_tokens_prev)
					# print(fixed)
					if self.DEBUG:
						te = time.perf_counter()
						logging.info(f'engine {idx} sampled tokens fixed in {te-ts:4f} seconds')
						ts = te
					# fixed = print(fixed)
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
				logging.info(f'engine {idx} generation steps: {my_generation_steps}')
				logging.info(f'engine {idx} total generation cycle time: {te-t_of_last_loop:4f}')
				logging.info(f'engine {idx} generation cycles per second: {1/(te-t_of_last_loop):4f}')
				ts = te
				t_of_last_loop = te
		print(f"generate thread {idx} exiting")

	def _processing_thread(self, idx):
		def add_to_slots(timestep, arr):
			n_free_slots = 0
			with self._slots_locks[idx]:
				for i, tok in enumerate(arr):
					if not self._slots[idx][i]['active']:
						n_free_slots += 1
						if self.DEBUG:
							print(f"slot {i} is not active")
						continue
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
							'end_reason': end_reason,
							'request_timestep': copy.deepcopy(self._slots[idx][i]['request_timestep']),
							'request_id': copy.deepcopy(self._slots[idx][i]['request_id']),
							'input_tokens': copy.deepcopy(self._slots[idx][i]['input_tokens']),
							'input_sequence': copy.deepcopy(self._slots[idx][i]['input_sequence']),
							'output_tokens': copy.deepcopy(self._slots[idx][i]['output_tokens']),
							'output_sequence': self.detokenize(copy.deepcopy(self._slots[idx][i]['output_tokens']), self._slots[idx][i]['request_config']['clean_up_tokenization_spaces'], self._slots[idx][i]['request_config']['skip_special_tokens']),
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
			n_events_set = 0
			if self._slots_freed_events[idx].is_set():
				n_events_set += 1
			n_allowed_to_set = n_free_slots - n_events_set
			n_set = 0
			if not self._slots_freed_events[idx].is_set() and n_allowed_to_set > 0:
				self._slots_freed_events[idx].set()
				n_allowed_to_set -= 1
				n_set += 1
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'processing {idx} set {n_set} slot free events in {te-ts:4f} seconds')
				ts = te

	def _tokenize_thread(self, idx):
		if self.DEBUG:
			ts = time.perf_counter()
			logging.info(f'tokenize thread {idx} ready')
		while self.live:
			request = self._request_queue.get()
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'tokenize thread {idx} got request in {te-ts:4f} seconds')
				ts = te
			out_tuple = self.tokenize(request)
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'tokenize thread {idx} tokenized in {te-ts:4f} seconds')
				ts = te
			while True:
				try:
					self._tokenized_request_queue.put(out_tuple, timeout=0.01)
					break
				except:
					if self.DEBUG:
						logging.info(f'tokenize thread {idx} failed to put tokenized request in queue')
					continue
			if self.DEBUG:
				te = time.perf_counter()
				logging.info(f'tokenize thread {idx} put tokenized request in {te-ts:4f} seconds')
				ts = te

	async def request(self, request, timeout = 0):
		try:
			if self._request_queue.full():
				if self.DEBUG:
					print(f"request {request['request_id']} timed out and was not put in queue")
				return None
			self._request_queue.put(request, timeout = timeout)
			if self.DEBUG:
				print(f"request {request['request_id']} put in queue")
		except queue.Full:
			if self.DEBUG:
				print(f"request {request['request_id']} timed out and was not put in queue")
			return None
		while True:
			responses_list = list(self._response_queue.queue)
			if len(responses_list) > 0:
				# if self.DEBUG:
				# 	print(f"request {request['request_id']} checking {len(responses_list)} responses")
				if responses_list[0]['request_id'] == request['request_id']:
					if self.DEBUG:
						print(f"request {request['request_id']} got response {responses_list[0]['request_id']}")
					return self._response_queue.get()
			await asyncio.sleep(1)
	
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

		self.n_tokenize_threads = server_config['n_tokenize_threads']
		assert self.n_tokenize_threads >= 1, "server config n_tokenize_threads must be >= 1"

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
		if self.prefill_request_get_timeout < 0.001:
			warnings.warn("""########################SERVER WARNING########################
SERVER CONFIG 'prefill_request_get_timeout' IS LESS THAN 0.001.
PERFORMANCE COULD BE LOWER THAN EXPECTED DUE TO GIL CONTENTION.""")

		self.max_one_time_insertion = server_config['max_one_time_insertion']
		assert self.max_one_time_insertion >= 1, "server configmax_one_time_insertion must be >= 1"

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
		assert self.sampling_strategy in ["greedy", "weighted", "top_k", "nucleus"], "server config sampling_strategy must be 'greedy', 'weighted', 'top_k', or 'nucleus'"
		
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
			return_tensors = "jax",
			)
		input_ids = tokenized.input_ids[0]
		attention_mask = tokenized.attention_mask[0]
		true_length = jnp.count_nonzero(attention_mask)
		nearest_length = take_nearest_length (
			self.tokenizer_config['possible_lengths'][:self.tokenizer_config['possible_lengths'].index(request_max_length) + 1],
			true_length,
		)
		input_ids = input_ids[:nearest_length]
		attention_mask = attention_mask[:nearest_length]
		true_length = jnp.count_nonzero(attention_mask)
		token_positions = jnp.arange(0, input_ids.shape[0])
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
		return input_ids, attention_mask, true_length, token_positions, formatted_request
	
	def detokenize(self, input_ids, clean_up_tokenization_spaces, skip_special_tokens):
		return self.tokenizer.decode (
			input_ids,
			clean_up_tokenization_spaces = clean_up_tokenization_spaces,
			skip_special_tokens = skip_special_tokens,
			)

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