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
		print(A)
		grid = nn.functional.affine_grid(A, size=volume.size(), align_corners=True)
				
		return nn.functional.grid_sample(volume, grid, mode=self.mode, padding_mode=self.padding_mode)