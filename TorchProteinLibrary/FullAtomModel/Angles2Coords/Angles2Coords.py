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


class Angles2CoordsFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	# @profile	
	@staticmethod
	def forward(ctx, input_angles_cpu, sequenceTensor, num_atoms, polymer_type, na_num_atoms):

		if polymer_type == 0:
			ctx.save_for_backward(input_angles_cpu, sequenceTensor)
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
												 na_num_atoms)

			if math.isnan(output_coords_cpu.sum()):
				raise(Exception('Angles2CoordsFunction: forward Nan'))

			ctx.mark_non_differentiable(output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms)
			return output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms

		if polymer_type == 1:
			ctx.save_for_backward(input_angles_cpu, sequenceTensor)
			input_angles_cpu = input_angles_cpu.contiguous()

			max_num_atoms = na_num_atoms
			batch_size = input_angles_cpu.size(0)
			output_coords_cpu = torch.zeros(batch_size, 3 * max_num_atoms, dtype=input_angles_cpu.dtype)
			output_chainnames_cpu = torch.zeros(batch_size, max_num_atoms, 1, dtype=torch.uint8).fill_(ord('A'))
			output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
			output_resnums_cpu = torch.zeros(batch_size, max_num_atoms, dtype=torch.int)
			output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

			print(sequenceTensor,
				 input_angles_cpu,
				 polymer_type,
				 na_num_atoms)

			_FullAtomModel.Angles2Coords_forward(sequenceTensor,
												 input_angles_cpu,
												 output_coords_cpu,
												 output_resnames_cpu,
												 output_resnums_cpu,
												 output_atomnames_cpu,
												 polymer_type,
												 na_num_atoms)

			if math.isnan(output_coords_cpu.sum()):
				raise(Exception('Angles2CoordsFunction: forward Nan'))

			ctx.mark_non_differentiable(output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms)
			return output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms
	
	# @profile
	@staticmethod 
	def backward(ctx, grad_atoms_cpu, *kwargs, polymer_type):
		# ATTENTION! It passes non-contiguous tensor
		grad_atoms_cpu = grad_atoms_cpu.contiguous()		
		input_angles_cpu, sequenceTensor = ctx.saved_tensors
		grad_angles_cpu = torch.zeros_like(input_angles_cpu)
				
		_FullAtomModel.Angles2Coords_backward(grad_atoms_cpu, grad_angles_cpu, sequenceTensor, input_angles_cpu, polymer_type)

		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		

		return grad_angles_cpu, None, None

class Angles2Coords(Module):
	def __init__(self, polymer_type=0, na_num_atoms=0):
		super(Angles2Coords, self).__init__()
		self.num_atoms = None
		self.polymer_type = polymer_type
		self.na_num_atoms = na_num_atoms

	def forward(self, input_angles_cpu, sequences, polymer_type, na_num_atoms):
		stringListTensor = convertStringList(sequences)
				
		self.num_atoms = []
		for seq in sequences:
			self.num_atoms.append(_FullAtomModel.getSeqNumAtoms(seq))
		num_atoms = torch.IntTensor(self.num_atoms)
		
		return Angles2CoordsFunction.apply(input_angles_cpu, stringListTensor, num_atoms, self.polymer_type, na_num_atoms)
