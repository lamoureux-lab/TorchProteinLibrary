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

		grid = torch.zeros(batch_size, size, size, size, 3, dtype=torch.float, device='cuda')
		_Volume.VolumeGenGrid(ctx.U, grid)
		
		return F.grid_sample(grad_output_volume, grid, mode=ctx.mode, padding_mode=ctx.padding_mode), None, None, None
		

class VolumeRotation(torch.nn.Module):
	"""
	coordinates rotation
	"""
	def __init__(self, mode='bilinear', padding_mode='zeros', fields=None):
		super(VolumeRotation, self).__init__()
		self.mode = mode
		self.padding_mode = padding_mode
		self.fields = fields
		

	def forward(self, input_volume, rotations):
		rot = VolumeRotationFunction.apply(input_volume, rotations, self.mode, self.padding_mode)
		
		if not self.fields is None:
			fields = []
			for interval in self.fields:
				#scalar field
				if interval[1] - interval[0] == 1:
					fields.append(origin_rot[:,interval[0],:,:,:])
				
				#vector field
				if interval[1] - interval[0] == 3:
					ix = interval[0]
					iy = interval[1]
					iz = interval[2]
					vec_x = origin_rot[:,ix,:,:,:].copy()
					vec_y = origin_rot[:,iy,:,:,:].copy()
					vec_z = origin_rot[:,iz,:,:,:].copy()
					rotations = rotations.unsqueeze(dim=3).unsqueeze(dim=4).unsqueeze(dim=5)
					origin_rot[:,ix,:,:,:] = rotations[:, 0, 0, :, :, :] * vec_x + rotations[:, 0, 1, :, :, :] * vec_y + rotations[:, 0, 2, :, :, :] * vec_z
					origin_rot[:,iy,:,:,:] = rotations[:, 1, 0, :, :, :] * vec_x + rotations[:, 1, 1, :, :, :] * vec_y + rotations[:, 1, 2, :, :, :] * vec_z
					origin_rot[:,iz,:,:,:] = rotations[:, 2, 0, :, :, :] * vec_x + rotations[:, 2, 1, :, :, :] * vec_y + rotations[:, 2, 2, :, :, :] * vec_z
					fields.append(origin_rot[:,interval[0]:interval[1],:,:,:])
					
			rot = torch.cat(fields, dim=1)

		return rot