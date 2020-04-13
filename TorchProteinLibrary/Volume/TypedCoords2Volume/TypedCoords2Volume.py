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
	def forward(ctx, input_coords_gpu, num_atoms_gpu, box_size, resolution):
		num_atoms_cpu = num_atoms_gpu.cpu()
		ctx.save_for_backward(input_coords_gpu, num_atoms_cpu, resolution)
		if len(input_coords_gpu.size())==2:
			batch_size = input_coords_gpu.size(0)
			volume_gpu = torch.zeros(batch_size, box_size, box_size, box_size, dtype=input_coords_gpu.dtype, device='cuda')
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', input_coords_gpu.size()) 
		
		_Volume.TypedCoords2Volume_forward(input_coords_gpu, volume_gpu, num_atoms_cpu, resolution.item())
		
		if math.isnan(volume_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: forward Nan'))	
		
		return volume_gpu
	
	@staticmethod
	def backward(ctx, grad_volume_gpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_volume_gpu = grad_volume_gpu.contiguous()
		input_coords_gpu, num_atoms_cpu, resolution = ctx.saved_tensors
		
		if len(grad_volume_gpu.size()) == 4:
			num_coords = input_coords_gpu.size(1)
			batch_size = grad_volume_gpu.size(0)
			grad_coords_gpu = torch.zeros(batch_size, num_coords, dtype=input_coords_gpu.dtype, device='cuda')
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', grad_volume_gpu.size()) 
		
		_Volume.TypedCoords2Volume_backward(grad_volume_gpu, grad_coords_gpu, input_coords_gpu, num_atoms_cpu, resolution.item())
		
		if math.isnan(grad_coords_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: backward Nan'))		
		
		return grad_coords_gpu, None, None, None

class TypedCoords2Volume(Module):
	"""
	Coordinated arranged in atom types function -> Volume
	"""
	def __init__(self, box_size=120, resolution=1.0):
		super(TypedCoords2Volume, self).__init__()
		self.box_size = box_size
		self.resolution = torch.tensor([resolution], dtype=torch.float)
						
	def forward(self, input_coords, num_atoms):
		batch_size = input_coords.size(0)
		num_types = input_coords.size(1)
		max_num_coords = input_coords.size(2)
		num_atom_types = num_atoms.size(1)
		
		input_coords = input_coords.view(batch_size*num_types, max_num_coords).contiguous()
		num_atoms = num_atoms.view(batch_size*num_atom_types).contiguous()

		volume = TypedCoords2VolumeFunction.apply(input_coords, num_atoms, self.box_size, self.resolution)

		return volume.view(batch_size, num_atom_types, self.box_size, self.box_size, self.box_size)
