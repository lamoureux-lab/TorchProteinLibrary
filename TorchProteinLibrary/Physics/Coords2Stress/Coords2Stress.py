import os
import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
import torch.nn.functional as F
import _Physics
import math
import numpy as np

class Coords2StressFunction(Function):
	@staticmethod
	def forward(ctx, coords, num_atoms, box_size, resolution):
		coords = coords.contiguous()
		num_atoms = num_atoms.contiguous()
		ctx.save_for_backward(coords, num_atoms)
		ctx.resolution = resolution

		batch_size = coords.size(0)
		

		return None

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		coords, num_atoms = ctx.saved_tensors
		
		num_coords = coords.size(1)
		gradInput = torch.zeros_like(coords)
						
		return None, None, None, None

class Coords2Stress(Module):
	def __init__(self, box_size=80, resolution=1.0):
		super(Coords2Stress, self).__init__()
		self.sigma2 = 1.0
			
	def get_distance_mat(self, coords, num_atoms):
		batch_size = coords.size(0)
		max_num_atoms = int(coords.size(1)/3)
		

		dist_mats = []
		for i in range(batch_size):
			num_at = num_atoms[i].item()
			this_coords = coords[i,:num_at*3].view(num_at, 3).contiguous()
			sep_mat = this_coords.unsqueeze(dim=1) - this_coords.unsqueeze(dim=0)
			dist_mat = torch.sqrt((sep_mat*sep_mat).sum(dim=2))
			dist_mat_pad = F.pad(dist_mat, (0, max_num_atoms - num_at, 0, max_num_atoms - num_at), 'constant', 0.0)
			dist_mats.append(dist_mat_pad)
		dist_mats = torch.stack(dist_mats, dim=0)
		return dist_mats
	
	def make_kirchoff(self, dist_mat, num_atoms):
		batch_size = dist_mat.size(0)
		max_num_atoms = int(dist_mat.size(1))
		for i in range(batch_size):
			num_at = num_atoms[i].item()
			for j in range(num_at):
				dist_mat[i,j,j] = -(dist_mat[i, j, :j].sum() + dist_mat[i, j, j:].sum())
		
		return dist_mat

	def forward(self, coords, num_atoms):
		dist_mat = self.get_distance_mat(coords, num_atoms)
		dist_mat = -torch.exp(-dist_mat/self.sigma2)
		kirchoff = self.make_kirchoff(dist_mat, num_atoms)
		return kirchoff

		
		