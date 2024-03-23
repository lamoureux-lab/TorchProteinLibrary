import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import math
import sys
import os
import numpy as np
import _FullAtomModel


def convertStringList(stringList):
	'''Converts list of strings to 0-terminated byte tensor'''
	maxlen = 0
	for string in stringList:
		string += '\0'
		if len(string)>maxlen:
			maxlen = len(string)
	ar = np.zeros( (len(stringList), maxlen), dtype=np.uint8)
	
	for i,string in enumerate(stringList):
		# npstring = np.fromstring(string, dtype=np.uint8)
		npstring = np.frombuffer(bytes(string,'ascii'), dtype=np.uint8)
		ar[i,:npstring.shape[0]] = npstring
	
	return torch.from_numpy(ar)

def convertString(string):
	'''Converts a string to 0-terminated byte tensor'''  
	return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))

def convert2str(tensor):
    return tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]


class Angles2CoordsFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	# @profile	
	@staticmethod
	def forward(ctx, input_angles_cpu, sequenceTensor, num_atoms, polymer_type, chain_names):

		if polymer_type == 0:
			ctx.save_for_backward(input_angles_cpu, sequenceTensor, torch.tensor(polymer_type, dtype=torch.int32), chain_names)
			input_angles_cpu = input_angles_cpu.contiguous()

			max_num_atoms = torch.max(num_atoms)
			batch_size = input_angles_cpu.size(0)
			output_coords_cpu = torch.zeros(batch_size, 3*max_num_atoms, dtype=input_angles_cpu.dtype)
			output_chainnames_cpu = torch.zeros(batch_size, max_num_atoms, 1, dtype=torch.uint8).fill_(ord('A'))
			output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
			output_resnums_cpu = torch.zeros(batch_size, max_num_atoms, dtype=torch.int)
			output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

			_FullAtomModel.Angles2Coords_forward(sequenceTensor,
												 input_angles_cpu,
												 output_coords_cpu,
												 output_resnames_cpu,
												 output_resnums_cpu,
												 output_atomnames_cpu,
												 polymer_type,
												 chain_names)

			if math.isnan(output_coords_cpu.sum()):
				raise(Exception('Angles2CoordsFunction: forward Nan'))

			ctx.mark_non_differentiable(output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms)
			return output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms

		if polymer_type == 1 or polymer_type == 2:
			ctx.save_for_backward(input_angles_cpu, sequenceTensor, torch.tensor(polymer_type, dtype=torch.int32), chain_names)
			input_angles_cpu = input_angles_cpu.contiguous()
			# print('num atoms', num_atoms)
			# print("a2c function poly 1 seq",str(convert2str(sequenceTensor))[2:-1])
			# max_num_atoms = (int(len(str(convert2str(sequenceTensor))) - 3) * 9) + (- 2) + (114) #for test [(seq - 3) * 6 for backbone length][(+ (-2)) for loss of phos group due to new chain] + 52 due to addition of Cyt & Thy rigid group without changing to getNum atoms
			max_num_atoms = num_atoms
			batch_size = input_angles_cpu.size(0)
			# print('batch size', batch_size)
			output_coords_cpu = torch.zeros(batch_size, 3 * max_num_atoms, dtype=input_angles_cpu.dtype)
			# output_chainnames_cpu = torch.zeros(batch_size, max_num_atoms, 1, dtype=torch.uint8).fill_(ord('A')) ##needs to be able to detect and write correct chain names
			output_chainnames_cpu = chain_names ##should be modified to detect new chains rather than just assign as chain_names arg
			output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
			output_resnums_cpu = torch.zeros(batch_size, max_num_atoms, dtype=torch.int)
			output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

			# print(sequenceTensor,
			# 	 input_angles_cpu,
			# 	 polymer_type,
			# 	 max_num_atoms)

			_FullAtomModel.Angles2Coords_forward(sequenceTensor,
												 input_angles_cpu,
												 output_coords_cpu,
												 output_resnames_cpu,
												 output_resnums_cpu,
												 output_atomnames_cpu,
												 polymer_type,
												 chain_names)

			if math.isnan(output_coords_cpu.sum()):
				raise(Exception('Angles2CoordsFunction: forward Nan'))

			ctx.mark_non_differentiable(output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms)
			# print("output_chainnames_cpu:", output_chainnames_cpu, "\n")
			return output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms
	
	# @profile
	@staticmethod 
	def backward(ctx, grad_atoms_cpu, *kwargs): ##, polymer_type=0, chain_names=torch.zeros(240, 1, dtype=torch.uint8).fill_(ord('A'))
		# ATTENTION! It passes non-contiguous tensor
		grad_atoms_cpu = grad_atoms_cpu.contiguous()		
		input_angles_cpu, sequenceTensor, polymer_type, chain_names = ctx.saved_tensors
		polymer_type_int = int(polymer_type)
		grad_angles_cpu = torch.zeros_like(input_angles_cpu)

		# print("input angles:", input_angles_cpu)
		# print("grad angles:", grad_angles_cpu)


		_FullAtomModel.Angles2Coords_backward(grad_atoms_cpu, grad_angles_cpu, sequenceTensor, input_angles_cpu, polymer_type_int, chain_names)

		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		

		return grad_angles_cpu, None, None, None, None

class Angles2Coords(Module):
	def __init__(self):
		super(Angles2Coords, self).__init__()
		self.num_atoms = None

	def forward(self, input_angles_cpu, sequences, chain_names, polymer_type=0):
		stringListTensor = convertStringList(sequences)
				
		# print(chain_names)
		self.num_atoms = []
		for seq in sequences:
			self.num_atoms.append(_FullAtomModel.getSeqNumAtoms(seq, polymer_type, chain_names))
		num_atoms = torch.IntTensor(self.num_atoms)

		return Angles2CoordsFunction.apply(input_angles_cpu, stringListTensor, num_atoms, polymer_type, chain_names)
