import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters, Coords2Elec
import _Volume


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

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
		self.p2c = PDB2CoordsUnordered()

class TestCoords2Stress_forward(TestCoords2Stress):

	def runTest(self):
        pass


if __name__=='__main__':
    unittest.main()