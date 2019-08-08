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
	def forward(ctx, input_angles_cpu, sequenceTensor, num_atoms):
		ctx.save_for_backward(input_angles_cpu, sequenceTensor)
		input_angles_cpu = input_angles_cpu.contiguous()
		
		max_num_atoms = torch.max(num_atoms)
		batch_size = input_angles_cpu.size(0)
		output_coords_cpu = torch.zeros(batch_size, 3*max_num_atoms, dtype=input_angles_cpu.dtype)
		output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
		output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

		_FullAtomModel.Angles2Coords_forward( 	sequenceTensor,
												input_angles_cpu, 
												output_coords_cpu, 
												output_resnames_cpu,
												output_atomnames_cpu)
		
		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: forward Nan'))
		
		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, num_atoms
	
	# @profile
	@staticmethod 
	def backward(ctx, grad_atoms_cpu, *kwargs):
		# ATTENTION! It passes non-contiguous tensor
		grad_atoms_cpu = grad_atoms_cpu.contiguous()		
		input_angles_cpu, sequenceTensor = ctx.saved_tensors
		grad_angles_cpu = torch.zeros_like(input_angles_cpu)
				
		_FullAtomModel.Angles2Coords_backward(grad_atoms_cpu, grad_angles_cpu, sequenceTensor, input_angles_cpu)

		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		

		return grad_angles_cpu, None, None

class Angles2Coords(Module):
	def __init__(self):
		super(Angles2Coords, self).__init__()
		self.num_atoms = None
		
	def forward(self, input_angles_cpu, sequences):
		stringListTensor = convertStringList(sequences)
				
		self.num_atoms = []
		for seq in sequences:
			self.num_atoms.append(_FullAtomModel.getSeqNumAtoms(seq))
		num_atoms = torch.IntTensor(self.num_atoms)
		
		return Angles2CoordsFunction.apply(input_angles_cpu, stringListTensor, num_atoms)