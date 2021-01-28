import torch
import torch.nn as nn
import math

import sys
import os

import _Volume

class VolumeRotation(nn.Module):
	"""
	Volume rotation by matrix
	Rotates around the volume center: [(box_size-1)/2,(box_size-1)/2,(box_size-1)/2]
	"""
	def __init__(self, mode='bilinear', padding_mode='zeros'):
		super(VolumeRotation, self).__init__()
		self.mode = mode
		self.padding_mode = padding_mode
		
	def forward(self, volume, R):
		batch_size = volume.size(0)
		num_features = volume.size(1)
		perm = torch.tensor([2,1,0], dtype=torch.long, device=R.device)
		R = R[:, :, perm]
		R = R[:, perm, :]
		TransMat = torch.zeros(batch_size, 3, device=R.device, dtype=R.dtype)
		A = torch.cat([R.transpose(1,2), TransMat.unsqueeze(dim=2)], dim=2)
		grid = nn.functional.affine_grid(A, size=volume.size(), align_corners=True)
				
		return nn.functional.grid_sample(volume, grid, mode=self.mode, padding_mode=self.padding_mode)


class VolumeRotationSE3(nn.Module):
	"""
	Fields in a volume rotation by matrix
	Rotates around the volume center: [(box_size-1)/2,(box_size-1)/2,(box_size-1)/2]
	"""
	def __init__(self, fields, **args):
		super(VolumeRotationSE3, self).__init__()
		self.vol_rotate = VolumeRotation(args)
		self.fields = fields

	def forward(self, volume, R):
		batch_size = volume.size(0)
		box_size = volume.size(-1)
		rotated_vol = self.vol_rotate(volume, R)
		idx_start = 0
		for m, l in self.fields:
			if m==0: continue
			idx_end = idx + m*(2*l+1)
			fields = rotated_vol[:, idx_start:idx_end, :, :, :].view(batch_size*m, 2*l+1, box_size, box_size, box_size)
			if l==0:
				pass
			elif l==1:
				fields = torch.einsum("aij,abjcde->abicde", R, volume)
			else:
				raise(Exception("Not implemented"))

			rotated_vol[:, idx_start:idx_end, :, :, :] = fields.view(batch_size, m*(2*l+1), box_size, box_size, box_size)
			idx_start = idx_end

		return rotated_vol
	