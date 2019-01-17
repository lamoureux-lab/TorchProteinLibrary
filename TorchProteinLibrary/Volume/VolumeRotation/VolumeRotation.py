import torch
import torch.nn.functional as F
from torch.autograd import Function
import math

import sys
import os

import _Volume

class VolumeRotationFunction(Function):
	"""
	Volume rotation
	"""
	@staticmethod
	def forward(ctx, input_volume, rotations, mode, padding_mode):
		batch_size = input_volume.size(0)
		num_features = input_volume.size(1)
		size = input_volume.size(2)
		ctx.mode = mode
		ctx.padding_mode = padding_mode

		ctx.U = rotations
		ctx.UT = torch.transpose(rotations, 1, 2).contiguous()
		grid = torch.zeros(batch_size, size, size, size, 3, dtype=torch.float, device='cuda')
		
		_Volume.VolumeGenGrid(ctx.UT, grid)
		return F.grid_sample(input_volume, grid, mode=ctx.mode, padding_mode=ctx.padding_mode)

	@staticmethod
	def backward(ctx, grad_output_volume):
		batch_size = grad_output_volume.size(0)
		num_features = grad_output_volume.size(1)
		size = grad_output_volume.size(2)

		grid = torch.zeros(batch_size, size, size, size, 3)
		_Volume.VolumeGenGrid(ctx.U, grid)
		
		return F.grid_sample(grad_output_volume, grid, mode=ctx.mode, padding_mode=ctx.padding_mode)
		

class VolumeRotation(torch.nn.Module):
	"""
	coordinates rotation
	"""
	def __init__(self, mode='bilinear', padding_mode='zeros'):
		super(VolumeRotation, self).__init__()
		self.mode = mode
		self.padding_mode = padding_mode
		

	def forward(self, input_volume, rotations):
		return VolumeRotationFunction.apply(input_volume, rotations, self.mode, self.padding_mode)