import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppTypedCoords2Volume
import math

import sys
import os

class TypedCoords2VolumeFunction(Function):
	"""
	coordinates -> coordinated arranged in atom types function
	"""
	def __init__(self, box_size=120, resolution=1.0):
		super(TypedCoords2VolumeFunction, self).__init__()
		self.box_size = box_size
		self.resolution = resolution
		self.num_atom_types = 11
				
	def forward(self, input_coords_cpu, num_atoms_of_type_cpu, offsets_cpu):
		
		if len(input_coords_cpu.size())==1:
			self.input_coords_gpu = torch.DoubleTensor(input_coords_cpu.size(0)).cuda().copy_(input_coords_cpu)
			self.num_atoms_of_type_gpu = torch.IntTensor(self.num_atom_types).cuda().copy_(num_atoms_of_type_cpu)
			self.offsets_gpu = torch.IntTensor(self.num_atom_types).cuda().copy_(offsets_cpu)
			volume_gpu = torch.FloatTensor(self.num_atom_types, self.box_size, self.box_size, self.box_size).cuda()
			
		elif len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			self.input_coords_gpu = torch.DoubleTensor(batch_size, input_coords_cpu.size(1)).cuda().copy_(input_coords_cpu)
			self.num_atoms_of_type_gpu = torch.IntTensor(batch_size, self.num_atom_types).cuda().copy_(num_atoms_of_type_cpu)
			self.offsets_gpu = torch.IntTensor(batch_size, self.num_atom_types).cuda().copy_(offsets_cpu)
			volume_gpu = torch.FloatTensor(batch_size, self.num_atom_types, self.box_size, self.box_size, self.box_size).cuda()
		
		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		volume_gpu.fill_(0.0)
		
		cppTypedCoords2Volume.TypedCoords2Volume_forward(self.input_coords_gpu, volume_gpu, self.num_atoms_of_type_gpu, self.offsets_gpu)

		if math.isnan(volume_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: forward Nan'))	

		return volume_gpu
			
	def backward(self, grad_volume_gpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_volume_gpu = grad_volume_gpu.contiguous()

		if len(grad_volume_gpu.size()) == 4:
			grad_coords_cpu = torch.DoubleTensor(self.input_coords_gpu.size(0))
			grad_coords_gpu = torch.DoubleTensor(self.input_coords_gpu.size(0)).cuda()
					
		elif len(grad_volume_gpu.size()) == 5:
			batch_size = grad_volume_gpu.size(0)
			grad_coords_cpu = torch.DoubleTensor(batch_size, self.input_coords_gpu.size(1))
			grad_coords_gpu = torch.DoubleTensor(batch_size, self.input_coords_gpu.size(1)).cuda()

		else:
			raise ValueError('TypedCoords2VolumeFunction: ', 'Incorrect input size:', grad_volume_gpu.size()) 
		
		cppTypedCoords2Volume.TypedCoords2Volume_backward(grad_volume_gpu, grad_coords_gpu, self.input_coords_gpu, self.num_atoms_of_type_gpu, self.offsets_gpu)
		
		if math.isnan(grad_coords_gpu.sum()):
			raise(Exception('TypedCoords2VolumeFunction: backward Nan'))		
		
		grad_coords_cpu.copy_(grad_coords_gpu)
		return grad_coords_cpu, None, None

class TypedCoords2Volume(Module):
	def __init__(self, box_size=120, resolution=1.0):
		super(TypedCoords2Volume, self).__init__()
		self.box_size = box_size
		self.resolution = resolution
		self.num_atom_types = 11
		
	def forward(self, input_coords_cpu, num_atoms_of_type_cpu, offsets_cpu):
		return TypedCoords2VolumeFunction(self.box_size, self.resolution)(input_coords_cpu, num_atoms_of_type_cpu, offsets_cpu)