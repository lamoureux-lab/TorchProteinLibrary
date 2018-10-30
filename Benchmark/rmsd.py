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

class Coords2RMSDBenchmark(ModuleBenchmark):
	def __init__(self, device='gpu', num_sequences=32, num_atoms=350):
		super().__init__(device)
		self.coords = torch.randn(num_sequences, 3*num_atoms, dtype=torch.double, device='cuda', requires_grad=True)
		self.ref = torch.randn(num_sequences, 3*num_atoms, dtype=torch.double, device='cuda', requires_grad=False)
		self.length = torch.zeros(num_sequences, dtype=torch.int, device='cuda').fill_(num_atoms)
		self.module = RMSD.Coords2RMSD().cuda()
				
	def prepare(self):
		self.coords.requires_grad_()
		self.ref.detach()
		self.rmsd = self.module(self.coords, self.ref, self.length)
		self.grad_rmsd = torch.ones(self.rmsd.size(), dtype=torch.double, device='cuda')
	
	def run_forward(self):
		self.coords.detach()
		# self.ref.detach()
		self.rmsd = self.module(self.coords, self.ref, self.length)
		# self.rmsd.detach()

	def run_backward(self):
		self.rmsd.backward(self.grad_rmsd)
		# self.rmsd.detach()

def test_length( interval=(60, 20, 300), num_measurements=1, output_filename='length.dat'):
	time = []
	direction = []
	length = []
	for n in range(num_measurements):
		for i in tqdm(range(interval[0], interval[2], interval[1])):
			mb = Coords2RMSDBenchmark(num_atoms=i)
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
							'NumAtoms': length
						})
	data.to_pickle(output_filename)

if __name__=='__main__':

	test_length(interval=(100, 100, 3000), output_filename='Data/RMSDTime.pkl', num_measurements=10)

	data = pd.read_pickle('Data/RMSDTime.pkl')
	sea.set_style("whitegrid")
	sea.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
	g1 = sea.relplot(x="NumAtoms", y="Time", hue='Direction', kind="line", style="Direction", height=6, aspect=1.3, markers=True, data=data)
	plt.ylabel("Time, ms")
	plt.xlabel("Number of atoms")
	sea.despine()
	# plt.show()
	plt.savefig("Fig/RMSDTime.png")