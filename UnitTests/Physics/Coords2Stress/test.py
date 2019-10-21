import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, Coords2Stress
import _Volume

import prody as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

import matplotlib.pylab as plt

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
			self.coords[0, 3*i + 1] = i*1.5 + 2.0
			self.coords[0, 3*i + 2] = i*1.5 + 2.0

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
		plt.show()


if __name__=='__main__':
	unittest.main()