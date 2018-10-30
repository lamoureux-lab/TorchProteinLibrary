from core import ModuleBenchmark
import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD
import numpy as np
from Bio.PDB.Polypeptide import aa1
import random
import seaborn as sea
from matplotlib import pylab as plt
import pandas as pd
from tqdm import tqdm
import pickle as pkl

def gen_rand_seq(seq_len=150):
	return ''.join([random.choice(aa1) for i in range(seq_len)])

class Angles2CoordsBenchmark(ModuleBenchmark):
	def __init__(self, device='cpu', num_sequences=32, seq_length=350):
		self.module = FullAtomModel.Angles2Coords()
		self.sequences = [gen_rand_seq(seq_length) for i in range(num_sequences)]
		# print(self.sequences)
		self.angles = torch.randn(len(self.sequences), 7, len(self.sequences[-1]), dtype=torch.double, device='cpu', requires_grad=True)
		
	def prepare(self):
		self.angles.requires_grad_()
		self.coords, res_names, atom_names, num_atoms = self.module(self.angles, self.sequences)
		self.grad_coords = torch.ones(self.coords.size(), dtype=torch.double, device='cpu')
	
	def run_forward(self):
		self.angles.detach()
		self.coords, res_names, atom_names, num_atoms = self.module(self.angles, self.sequences)

	def run_backward(self):
		self.coords.backward(self.grad_coords)


def test_length( interval=(60, 20, 300), num_measurements=1, output_filename='length.dat'):
	time = []
	direction = []
	length = []
	for n in range(num_measurements):
		for i in tqdm(range(interval[0], interval[2], interval[1])):
			mb = Angles2CoordsBenchmark(seq_length=i)
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

	# test_length(interval=(60, 20, 700), output_filename='Data/FullAtomModelTime.pkl', num_measurements=10)
	
	data = pd.read_pickle('Data/FullAtomModelTime.pkl')
	sea.set_style("whitegrid")
	sea.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
	g1 = sea.relplot(x="Length", y="Time", hue='Direction', kind="line", style="Direction", height=6, aspect=1.3, markers=True, data=data)
	plt.ylabel("Time, ms")
	plt.xlabel("Sequence length, amino-acids")
	sea.despine()
	# plt.show()
	plt.savefig("Fig/FullAtomModelTime.png")
