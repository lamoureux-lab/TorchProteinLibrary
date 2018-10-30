import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD
import numpy as np

class ModuleBenchmark:
	def __init__(self, device='cpu'):
		self.module = None
		if device=='cpu':
			self.use_cuda=False
		else:
			self.use_cuda=True

	def prepare(self):
		pass
	
	def run_forward(self):
		pass

	def run_backward(self):
		pass

	def measure_forward(self):
		self.prepare()
		with profiler.profile(use_cuda=self.use_cuda) as prof:
			self.run_forward()
		dur = 0.0
		threads = set([])
		if self.use_cuda:
			for evt in prof.function_events:
				threads.add(evt.thread)
				for k in evt.kernels:
					dur += k.interval.elapsed_us()
		else:
			for evt in prof.function_events:
				threads.add(evt.thread)
				dur += evt.cpu_interval.elapsed_us()
		# print(len(threads))
		return dur


	def measure_backward(self):        
		self.prepare()
		with profiler.profile(use_cuda=self.use_cuda) as prof:
			self.run_backward()
		dur = 0.0
		threads = set([])
		if self.use_cuda:
			for evt in prof.function_events:
				threads.add(evt.thread)
				for k in evt.kernels:
					dur += k.interval.elapsed_us()
		else:
			for evt in prof.function_events:
				threads.add(evt.thread)
				dur += evt.cpu_interval.elapsed_us()
		# print(len(threads))
		return dur
		
	