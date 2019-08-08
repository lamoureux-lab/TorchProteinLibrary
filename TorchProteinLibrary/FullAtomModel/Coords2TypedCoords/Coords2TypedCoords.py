import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _FullAtomModel
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
			output_coords_cpu = torch.zeros(batch_size, 3*max_num_atoms, dtype=input_coords_cpu.dtype)
			num_atoms_of_type = torch.zeros(batch_size, num_atom_types, dtype=torch.int)
			offsets = torch.zeros(batch_size, num_atom_types, dtype=torch.int)
			ctx.atom_indexes = torch.zeros(batch_size, max_num_atoms, dtype=torch.int)
			
		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		_FullAtomModel.Coords2TypedCoords_forward(   input_coords_cpu, input_resnames, input_atomnames, num_atoms,
													output_coords_cpu, num_atoms_of_type, offsets, ctx.atom_indexes)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: forward Nan'))	
		
		ctx.save_for_backward(num_atoms_of_type, offsets)
		return output_coords_cpu, num_atoms_of_type, offsets
	
	@staticmethod
	def backward(ctx, grad_typed_coords_cpu, *kwargs):
		# ATTENTION! It passes non-contiguous tensor
		grad_typed_coords_cpu = grad_typed_coords_cpu.contiguous()
		num_atom_types = 11
		num_atoms_of_type, offsets = ctx.saved_tensors

		if len(grad_typed_coords_cpu.size()) == 2:
			grad_coords_cpu = torch.zeros(grad_typed_coords_cpu.size(0), grad_typed_coords_cpu.size(1), dtype=grad_typed_coords_cpu.dtype)
		else:
			raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		_FullAtomModel.Coords2TypedCoords_backward(	grad_typed_coords_cpu, grad_coords_cpu, 
													num_atoms_of_type, 
													offsets, 
													ctx.atom_indexes)
		
		if math.isnan(grad_coords_cpu.sum()):
			raise(Exception('Coords2TypedCoordsFunction: backward Nan'))		
		
		return Variable(grad_coords_cpu), None, None, None

class Coords2TypedCoords(Module):
	def __init__(self):
		super(Coords2TypedCoords, self).__init__()
				
	def forward(self, input_coords_cpu, input_resnames, input_atomnames, num_atoms):
		return Coords2TypedCoordsFunction.apply(input_coords_cpu, input_resnames, input_atomnames, num_atoms)