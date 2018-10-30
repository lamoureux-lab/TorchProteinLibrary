from core import ModuleBenchmark
import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD

import numpy as np
import seaborn as sea
from matplotlib import pylab as plt
import pandas as pd
from tqdm import tqdm
import pickle as pkl

class LSTMBenchmark(ModuleBenchmark):
	def __init__(self, device='gpu', num_sequences=32, input_length=128, input_size=128, hidden_size=256, num_layers=1):
		super().__init__(device)
		self.input = torch.randn(num_sequences, input_length, input_size, dtype=torch.float, device='cuda', requires_grad=True)
		self.module = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).cuda()
				
	def prepare(self):
		self.input.requires_grad_()
		self.output, _ = self.module(self.input)
		self.grad_output = torch.ones(self.output.size(), dtype=torch.float, device='cuda')
	
	def run_forward(self):
		self.input.detach()
		self.output, _ = self.module(self.input)

	def run_backward(self):
		self.output.backward(self.grad_output)

def test_length( interval=(60, 20, 300), num_measurements=1, output_filename='length.dat'):
	time = []
	direction = []
	length = []
	for n in range(num_measurements):
		for i in tqdm(range(interval[0], interval[2], interval[1])):
			mb = LSTMBenchmark(input_length=i)
			dur_fwd = mb.measure_forward()
			time.append(dur_fwd/1000.0)
			direction.append('Forward')
			length.append(i)
			dur_bwd = mb.measure_backward()
			time.append(dur_bwd/1000.0)
			direction.append('Backward')
			length.append(i)
	
	data = pd.DataFrame({  'Time': time, 
							'Direction': direction, 
							'Length': length
						})
	data.to_pickle(output_filename)

if __name__=='__main__':

	test_length(interval=(60, 20, 700), output_filename='Data/LSTMTime.pkl', num_measurements=10)

	data = pd.read_pickle('Data/LSTMTime.pkl')
	sea.set_style("whitegrid")
	sea.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
	g1 = sea.relplot(x="Length", y="Time", hue='Direction', kind="line", style="Direction", height=6, aspect=1.3, markers=True, data=data)
	plt.ylabel("Time, ms")
	plt.xlabel("Sequence length, amino-acids")
	sea.despine()
	# plt.show()
	plt.savefig("Fig/LSTMTime.png")