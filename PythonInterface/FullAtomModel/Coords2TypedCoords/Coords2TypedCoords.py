import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppCoords2TypedCoords
import math

import sys
import os

class Coords2TypedCoordsFunction(Function):
	"""
	coordinates -> coordinated arranged in atom types function
	"""
	def __init__(self, num_atoms, num_atom_types=11):
		super(Coords2TypedCoordsFunction, self).__init__()
		self.num_atoms = num_atoms
		self.num_atom_types = num_atom_types
				
	def forward(self, input_coords_cpu, input_resnames, input_atomnames):
		
		if len(input_coords_cpu.size())==1:
			output_coords_cpu = torch.DoubleTensor(3*self.num_atoms)
			self.num_atoms_of_type = torch.IntTensor(self.num_atom_types)
			self.offsets = torch.IntTensor(self.num_atom_types)
			self.atom_indexes = torch.IntTensor(self.num_atoms)
			
		elif len(input_angles_cpu.size())==2:
			raise(Exception('Not implemented'))

		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		output_coords_cpu.fill_(0.0)
		self.num_atoms_of_type.fill_(0)
		self.offsets.fill_(0)
		self.atom_indexes.fill_(0)
		cppCoords2TypedCoords.Coords2TypedCoords_forward(    input_coords_cpu, input_resnames, input_atomnames,
															output_coords_cpu, self.num_atoms_of_type, self.offsets, self.atom_indexes)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: forward Nan'))	

		return output_coords_cpu, self.num_atoms_of_type, self.offsets
			
	def backward(self, grad_typed_coords_cpu, *kwargs):
		# ATTENTION! It passes non-contiguous tensor
		grad_typed_coords_cpu = grad_typed_coords_cpu.contiguous()

		if len(grad_typed_coords_cpu.size()) == 1:
			grad_coords_cpu = torch.DoubleTensor(grad_typed_coords_cpu.size(0))
					
		elif len(grad_typed_coords_cpu.size()) == 2:
			raise(Exception('Not implemented'))

		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		cppCoords2TypedCoords.Coords2TypedCoords_backward(grad_typed_coords_cpu, grad_coords_cpu, self.num_atoms_of_type, self.offsets, self.atom_indexes)
		
		if math.isnan(grad_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: backward Nan'))		
		
		return grad_coords_cpu, None, None

class Coords2TypedCoords(Module):
	def __init__(self, num_atoms, num_atom_types=11):
		super(Coords2TypedCoords, self).__init__()
		self.num_atoms = num_atoms
		self.num_atom_types = num_atom_types
		
	def forward(self, input_coords_cpu, input_resnames, input_atomnames):
		return Coords2TypedCoordsFunction(self.num_atoms, self.num_atom_types)(input_coords_cpu, input_resnames, input_atomnames)