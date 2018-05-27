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
			
	@staticmethod
	def forward(ctx, input_coords_cpu, input_resnames, input_atomnames, num_atoms):
		max_num_atoms = torch.max(num_atoms)
		num_atom_types = 11
			
		if len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			output_coords_cpu = torch.DoubleTensor(batch_size, 3*max_num_atoms)
			num_atoms_of_type = torch.IntTensor(batch_size, num_atom_types)
			offsets = torch.IntTensor(batch_size, num_atom_types)
			ctx.atom_indexes = torch.IntTensor(batch_size, max_num_atoms)
			
		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		output_coords_cpu.fill_(0.0)
		num_atoms_of_type.fill_(0)
		offsets.fill_(0)
		ctx.atom_indexes.fill_(0)
		cppCoords2TypedCoords.Coords2TypedCoords_forward(    input_coords_cpu, input_resnames, input_atomnames, num_atoms,
															output_coords_cpu, num_atoms_of_type, offsets, ctx.atom_indexes)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: forward Nan'))	
		
		ctx.save_for_backward(num_atoms_of_type, offsets)
		return output_coords_cpu, num_atoms_of_type, offsets
	
	@staticmethod
	def backward(self, grad_typed_coords_cpu, *kwargs):
		# ATTENTION! It passes non-contiguous tensor
		grad_typed_coords_cpu = grad_typed_coords_cpu.contiguous()
		num_atom_types = 11
		num_atoms_of_type, offsets = ctx.saved_tensors

		if len(grad_typed_coords_cpu.size()) == 2:
			grad_coords_cpu = torch.DoubleTensor(grad_typed_coords_cpu.size(0), grad_typed_coords_cpu.size(1))

		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		cppCoords2TypedCoords.Coords2TypedCoords_backward(	grad_typed_coords_cpu, grad_coords_cpu, 
															num_atoms_of_type, offsets, ctx.atom_indexes)
		
		if math.isnan(grad_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: backward Nan'))		
		
		return grad_coords_cpu, None, None

class Coords2TypedCoords(Module):
	def __init__(self):
		super(Coords2TypedCoords, self).__init__()
				
	def forward(self, input_coords_cpu, input_resnames, input_atomnames, num_atoms):
		return Coords2TypedCoordsFunction.apply(input_coords_cpu, input_resnames, input_atomnames, num_atoms)