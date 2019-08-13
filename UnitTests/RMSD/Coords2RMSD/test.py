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
	def setUp(self):
		self.a2c = Angles2Coords()
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.device = 'cuda'
		self.rmsd = Coords2RMSD()


class TestCoords2RMSDForward(TestCoords2RMSD):
	def runTest(self):
		length = 32
		batch_size = 16
		num_aa = torch.zeros(batch_size, dtype=torch.int, device='cpu').random_(int(length/2), length)
		sequences = [''.join(['G' for i in range(num_aa[j].item())]) for j in range(batch_size)]
		fa_angles = torch.randn(batch_size, 8, length, dtype=torch.double, device='cpu')
		coords_fa, res_names, atom_names, num_atoms = self.a2c(fa_angles, sequences)

		R = getRandomRotation(batch_size)
		T = torch.randn(batch_size, 3, dtype=torch.double, device='cpu')
		rot_coords_fa = self.rotate(coords_fa, R, num_atoms)
		tra_rot_coords_fa = self.translate(rot_coords_fa, T, num_atoms)

		cuda_tra_rot_coords_fa = tra_rot_coords_fa.to(device='cuda', dtype=torch.double)
		cuda_coords_fa = coords_fa.to(device='cuda', dtype=torch.double)
		# num_atoms = num_atoms.to(device='cuda', dtype=torch.int)

		min_rmsd = self.rmsd(cuda_tra_rot_coords_fa, cuda_coords_fa, num_atoms)
		print(min_rmsd)

if __name__=='__main__':
	unittest.main()

