import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel.Angles2Coords import Angles2Coords
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, CoordsRotate, getRandomRotation
from TorchProteinLibrary.RMSD import Coords2RMSD

class TestCoords2RMSD(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 5
	batch_size = 16
	length = 32
	max_num_atoms = 16
	eps=1e-06 
	atol=1e-05 
	rtol=0.001
	msg = "Testing Coords2RMSD"
	def setUp(self):
		print(self.msg, self.device, self.dtype)
		self.a2c = Angles2Coords()
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.rmsd = Coords2RMSD()


class TestCoords2RMSDForward(TestCoords2RMSD):
	msg = "Testing Coords2RMSDForward:"
	def runTest(self):
		num_aa = torch.zeros(self.batch_size, dtype=torch.int, device=self.device).random_(int(self.length/2), self.length)
		sequences = [''.join(['G' for i in range(num_aa[j].item())]) for j in range(self.batch_size)]
		fa_angles = torch.randn(self.batch_size, 8, self.length, dtype=torch.double, device='cpu')
		coords_fa, res_names, atom_names, num_atoms = self.a2c(fa_angles, sequences)


		R = getRandomRotation(self.batch_size)
		T = torch.randn(self.batch_size, 3, dtype=torch.double, device='cpu')
		rot_coords_fa = self.rotate(coords_fa, R, num_atoms)
		tra_rot_coords_fa = self.translate(rot_coords_fa, T, num_atoms)
		
		tra_rot_coords_fa = tra_rot_coords_fa.to(device=self.device, dtype=self.dtype)
		coords_fa = coords_fa.to(device=self.device, dtype=self.dtype)
		num_atoms = num_atoms.to(device=self.device, dtype=torch.int)
		min_rmsd = self.rmsd(tra_rot_coords_fa, coords_fa, num_atoms)
		for i in range(self.batch_size):
			self.assertAlmostEqual(min_rmsd[i].item(), 0.0, places=self.places)

class TestCoords2RMSDForward_CPUFloat(TestCoords2RMSDForward):
	device = 'cpu'
	dtype = torch.float
	places = 5

class TestCoords2RMSDForward_CUDADouble(TestCoords2RMSDForward):
	device = 'cuda'
	dtype = torch.double
	places = 5

class TestCoords2RMSDForward_CUDAFloat(TestCoords2RMSDForward):
	device = 'cuda'
	dtype = torch.float
	places = 5


class TestCoords2RMSDBackward(TestCoords2RMSD):
	eps=1e-06 
	atol=1e-05 
	rtol=1e-05 
	msg = "Testing Coords2RMSDBackward:"
	batch_size = 1
	def runTest(self):
		coords_src = torch.randn(self.batch_size, self.max_num_atoms*3, dtype=self.dtype, device=self.device).requires_grad_()
		coords_dst = torch.randn(self.batch_size, self.max_num_atoms*3, dtype=self.dtype, device=self.device)
		num_atoms = torch.zeros(self.batch_size, dtype=torch.int, device=self.device).random_(int(self.max_num_atoms/2), self.max_num_atoms)
		
		result = torch.autograd.gradcheck(self.rmsd, (coords_src, coords_dst, num_atoms), self.eps, self.atol, self.rtol)
		self.assertTrue(result)

class TestCoords2RMSDBackward_CUDADouble(TestCoords2RMSDBackward):
	device = 'cuda'
	dtype = torch.double

class TestCoords2RMSDBackward_CPUFloat(TestCoords2RMSDBackward):
	eps=1e-04 
	atol=1e-03 
	rtol=1e-03 
	device = 'cpu'
	dtype = torch.float

class TestCoords2RMSDBackward_CUDAFloat(TestCoords2RMSDBackward):
	eps=1e-04 
	atol=1e-03 
	rtol=1e-03 
	device = 'cuda'
	dtype = torch.float



if __name__=='__main__':
	unittest.main()

