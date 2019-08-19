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
	def forward(ctx, input_coords_gpu, num_atoms_of_type_gpu, offsets_gpu, box_size, resolution):
		ctx.save_for_backward(input_coords_gpu, num_atoms_of_type_gpu, offsets_gpu, resolution)
		num_atom_types = 11
		if len(input_coords_gpu.size())==2:
			batch_size = input_coords_gpu.size(0)
			volume_gpu = torch.zeros(batch_size, num_atom_types, box_size, box_size, box_size, dtype=input_coords_gpu.dtype, device='cuda')
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', input_coords_gpu.size()) 
		
		_Volume.TypedCoords2Volume_forward(input_coords_gpu, volume_gpu, num_atoms_of_type_gpu, offsets_gpu, resolution.item())
		
		if math.isnan(volume_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: forward Nan'))	

		return volume_gpu
	
	@staticmethod
	def backward(ctx, grad_volume_gpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_volume_gpu = grad_volume_gpu.contiguous()
		input_coords_gpu, num_atoms_of_type_gpu, offsets_gpu, resolution = ctx.saved_tensors
		
		if len(grad_volume_gpu.size()) == 5:
			num_coords = input_coords_gpu.size(1)
			batch_size = grad_volume_gpu.size(0)
			grad_coords_gpu = torch.zeros(batch_size, num_coords, dtype=input_coords_gpu.dtype, device='cuda')
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', grad_volume_gpu.size()) 
		
		_Volume.TypedCoords2Volume_backward(grad_volume_gpu, grad_coords_gpu, input_coords_gpu, num_atoms_of_type_gpu, offsets_gpu, resolution.item())
		
		if math.isnan(grad_coords_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: backward Nan'))		
		
		return grad_coords_gpu, None, None, None, None

class TypedCoords2Volume(Module):
	"""
	Coordinated arranged in atom types function -> Volume
	"""
	def __init__(self, box_size=120, resolution=1.0):
		super(TypedCoords2Volume, self).__init__()
		self.box_size = box_size
		self.resolution = torch.tensor([resolution], dtype=torch.float)
						
	def forward(self, input_coords_cpu, num_atoms_of_type_cpu, offsets_cpu):
		return TypedCoords2VolumeFunction.apply(input_coords_cpu, num_atoms_of_type_cpu, offsets_cpu, self.box_size, self.resolution)