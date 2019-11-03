import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params#, Coords2Stress
import _Volume

import prody as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

import matplotlib.pylab as plt

from torch import nn
import torch.nn.functional as F
class Coords2Stress(nn.Module):
	def __init__(self, box_size=80, resolution=1.0):
		super(Coords2Stress, self).__init__()
		self.sigma2 = 1.0
			
	def get_sep_mat(self, coords, num_atoms):
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

	def get_hessian(self, sep_mat, num_atoms):
		batch_size = sep_mat.size(0)
		max_num_atoms = sep_mat.size(1)

		dist_mat = torch.sqrt((sep_mat*sep_mat).sum(dim=3) + 1e-5)
		dist_mat = dist_mat.unsqueeze(dim=3).unsqueeze(dim=4)
		dist2_mat = dist_mat*dist_mat#*dist_mat*dist_mat
		hessian = (sep_mat.unsqueeze(dim=3)) * (sep_mat.unsqueeze(dim=4))
		hessian = -hessian / dist2_mat

		# hessian = torch.einsum("ijklm->ijjlm",hessian)

		for i in range(batch_size):
			num_at = num_atoms[i].item()
			for j in range(num_at):
				hessian[i,j,j,:,:] = -(hessian[i,j,:,:,:].sum(dim=0))

		hessian = hessian.transpose(2,3).contiguous()	
		return hessian.view(batch_size,max_num_atoms*3,max_num_atoms*3).contiguous()

	def forward(self, coords, num_atoms):
		batch_size = coords.size(0)
		max_coords = coords.size(1)

		sep_mat = self.get_sep_mat(coords, num_atoms)
		hessian = self.get_hessian(sep_mat, num_atoms)
		
		hessian_inverse = []
		for i in range(batch_size):
			num_coords = 3*num_atoms[i].item()
			Hinv = torch.pinverse(hessian[i,:num_coords,:num_coords])
			F.pad(Hinv, (0, max_coords - num_coords, 0, max_coords - num_coords), 'constant', 0.0)
			hessian_inverse.append(Hinv)

		hessian_inverse = torch.stack(hessian_inverse, dim=0)
		coords_displ = torch.einsum("ijj->ij",hessian_inverse)
		
		return hessian
		

class TestCoords2Stress(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 7
	batch_size = 16
	max_num_atoms = 30
	eps=1e-06 
	atol=1e-05 
	rtol=0.001
	msg = "Testing stress tensor computation"

	def setUp(self):
		num_at = 10
		self.coords = torch.randn(1, num_at*3, dtype=torch.float, device=self.device)
		self.num_atoms = torch.tensor([num_at], dtype=torch.int, device=self.device)
		for i in range(num_at):
			self.coords[0, 3*i + 0] = i*2.0 + 2.0
			self.coords[0, 3*i + 1] = 0.0#i*1.5 + 2.0
			self.coords[0, 3*i + 2] = 0.0#i*1.5 + 2.0

		self.coords2stress = Coords2Stress()
		
		
		
		

class TestCoords2Stress_forward(TestCoords2Stress):

	def runTest(self):
		# pdb = pd.parsePDB('test.pdb')
		# atoms = pdb.select('protein')
		# gnm = pd.GNM('Test')
		# gnm.buildKirchhoff(atoms)
		# gnm.calcModes()
		# pd.showContactMap(gnm)
		# print(gnm.getVariances())
		# print(gnm.getCutoff())
		# print(gnm.getGamma())
		# print(gnm.getCovariance().round(2))
		
		dist_mat = self.coords2stress(self.coords, self.num_atoms)
		plt.imshow(dist_mat[0,:,:].numpy())
		plt.colorbar()
		plt.show()


if __name__=='__main__':
	unittest.main()