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

class Coords2Stress(nn.Module):
	def __init__(self, box_size=80, resolution=1.0):
		super(Coords2Stress, self).__init__()
		self.sigma2 = 1.0
			
	def get_sep_mat(self, coords, num_atoms):
		"""
		Computes separation matrix: all coordinates minus all coordinates
		"""

		batch_size = coords.size(0)
		max_num_atoms = int(coords.size(1)/3)
		
		sep_mats = []
		for i in range(batch_size):
			num_at = num_atoms[i].item()
			this_coords = coords[i,:num_at*3].view(num_at, 3).contiguous()
			sep_mat = this_coords.unsqueeze(dim=1) - this_coords.unsqueeze(dim=0)
			sep_mat_pad = F.pad(sep_mat, (0, 3*max_num_atoms - 3*num_at, 0, 3*max_num_atoms - 3*num_at), 'constant', 0.0)
			sep_mats.append(sep_mat_pad)
		
		sep_mats = torch.stack(sep_mats, dim=0)
		return sep_mats

	def get_hessian(self, sep_mat, num_atoms, cutoff = 7.0):
		"""
		Computes Hessian of the interaction potential. Interaction is cut off using gaussian
		"""

		batch_size = sep_mat.size(0)
		max_num_atoms = sep_mat.size(1)

		dist_mat = torch.sqrt((sep_mat*sep_mat).sum(dim=3) + 1e-5)
		dist_mat = dist_mat.unsqueeze(dim=3).unsqueeze(dim=4)
		dist2_mat = dist_mat*dist_mat
		hessian = (sep_mat.unsqueeze(dim=3)) * (sep_mat.unsqueeze(dim=4))
		hessian = -hessian * torch.exp(-dist2_mat/(cutoff*cutoff)) / dist2_mat

		for i in range(batch_size):
			num_at = num_atoms[i].item()
			for j in range(num_at):
				hessian[i,j,j,:,:] = -(hessian[i,j,:,:,:].sum(dim=0))

		hessian = hessian.transpose(2,3).contiguous()	
		return hessian.view(batch_size,max_num_atoms*3,max_num_atoms*3).contiguous()

	def get_displacements(self, hessian, num_atoms):
		"""
		Computes larges eigenvector and returns 3*eigvec*eigval. This corresponds to
		the displacements of atoms in [kBT/gamma] units
		"""

		batch_size = hessian.size(0)
		max_num_coords = hessian.size(1)

		eigvecs, eigvals = [], []
		for i in range(batch_size):
			num_coords = 3*num_atoms[i].item()
			eigval, eigvec = torch.symeig(hessian[i,:num_coords, :num_coords], eigenvectors=True)
			
			if (torch.abs(eigval[-1]) > torch.abs(eigval[0])):
				eigvec = eigvec[-1,:]
				eigval = eigval[-1]
			else:
				eigvec = eigvec[0,:]
				eigval = eigval[0]
			
			eigvals.append(eigval)
			eigvec = F.pad(eigvec, (0, max_num_coords - num_coords), 'constant', 0.0)
			eigvecs.append(eigvec)

		eigvecs = torch.stack(eigvecs, dim=0)
		eigvals = torch.stack(eigvals, dim=0)

		return 3*eigvecs*eigvals


	def forward(self, coords, num_atoms):
		batch_size = coords.size(0)
		max_coords = coords.size(1)

		sep_mat = self.get_sep_mat(coords, num_atoms)
		hessian = self.get_hessian(sep_mat, num_atoms)
		displacements = self.get_displacements(hessian, num_atoms)
		
		return hessian, displacements

		
		