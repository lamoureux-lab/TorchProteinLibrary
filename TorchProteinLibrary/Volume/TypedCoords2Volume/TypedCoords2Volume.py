import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Volume
import math

import sys
import os

class TypedCoords2VolumeFunction(Function):
	
	@staticmethod
	def forward(ctx, input_coords_gpu, num_atoms_gpu, box_size, resolution, num_neighbours):
		num_atoms_cpu = num_atoms_gpu.cpu()
		ctx.save_for_backward(input_coords_gpu, num_atoms_cpu, resolution, num_neighbours)
		if len(input_coords_gpu.size())==2:
			batch_size = input_coords_gpu.size(0)
			volume_gpu = torch.zeros(batch_size, box_size, box_size, box_size, dtype=input_coords_gpu.dtype, device=input_coords_gpu.device)
			max_num_atoms = torch.max(num_atoms_cpu).item()
			num_neighbour_cells = (2*num_neighbours.item()+1)**3
			HASH_EMPTY = 2147483647
			sortedPos = torch.zeros(batch_size, num_neighbour_cells*max_num_atoms*3, dtype=input_coords_gpu.dtype, device=input_coords_gpu.device)
			particleHash = torch.zeros(batch_size, 2, num_neighbour_cells*max_num_atoms, dtype=torch.int64, device=input_coords_gpu.device).fill_(HASH_EMPTY)
			particleIndex = torch.zeros(batch_size, 2, num_neighbour_cells*max_num_atoms, dtype=torch.int64, device=input_coords_gpu.device).fill_(HASH_EMPTY)
			cellStart = torch.zeros(batch_size, box_size*box_size*box_size, dtype=torch.int64, device=input_coords_gpu.device).fill_(HASH_EMPTY)
			cellStop = torch.zeros(batch_size, box_size*box_size*box_size, dtype=torch.int64, device=input_coords_gpu.device).fill_(HASH_EMPTY)
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', input_coords_gpu.size()) 
		
		_Volume.TypedCoords2Volume_forward(	input_coords_gpu, volume_gpu, num_atoms_cpu, 
											resolution.item(), num_neighbours.item(),
											particleHash, particleIndex, cellStart, cellStop, sortedPos)
		
		if math.isnan(volume_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: forward Nan'))	
		
		return volume_gpu
	
	@staticmethod
	def backward(ctx, grad_volume_gpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_volume_gpu = grad_volume_gpu.contiguous()
		input_coords_gpu, num_atoms_cpu, resolution, num_neighbours = ctx.saved_tensors
		
		if len(grad_volume_gpu.size()) == 4:
			num_coords = input_coords_gpu.size(1)
			batch_size = grad_volume_gpu.size(0)
			grad_coords_gpu = torch.zeros(batch_size, num_coords, dtype=input_coords_gpu.dtype, device=input_coords_gpu.device)
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', grad_volume_gpu.size()) 
		
		_Volume.TypedCoords2Volume_backward(grad_volume_gpu, grad_coords_gpu, input_coords_gpu, num_atoms_cpu, 
											resolution.item(), num_neighbours.item())
		
		if math.isnan(grad_coords_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: backward Nan'))		
		
		return grad_coords_gpu, None, None, None, None

class TypedCoords2Volume(Module):
	"""
	Coordinated arranged in atom types function -> Volume
	"""
	def __init__(self, box_size=120, resolution=1.0, num_neighbours=2):
		super(TypedCoords2Volume, self).__init__()
		self.box_size = box_size
		self.resolution = torch.tensor([resolution], dtype=torch.float)
		self.num_neighbours = torch.tensor([num_neighbours], dtype=torch.int)
						
	def forward(self, input_coords, num_atoms):
		batch_size = input_coords.size(0)
		num_types = input_coords.size(1)
		max_num_coords = input_coords.size(2)
		num_atom_types = num_atoms.size(1)
		
		input_coords = input_coords.view(batch_size*num_types, max_num_coords).contiguous()
		num_atoms = num_atoms.view(batch_size*num_atom_types).contiguous()

		volume = TypedCoords2VolumeFunction.apply(input_coords, num_atoms, self.box_size, self.resolution, self.num_neighbours)

		return volume.view(batch_size, num_atom_types, self.box_size, self.box_size, self.box_size)
