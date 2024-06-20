import sys
import os
import torch
import numpy as np
import unittest

import TorchProteinLibrary
from TorchProteinLibrary.Volume import CoordsSelect

class TestCoordsSelect(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 4
	max_num_atoms = 16
	box_size_bin = 40
	box_size_ang = 120
	num_features = 12
	eps=1e-03 
	atol=1e-05 
	rtol=0.001
	msg = "Testing CoordsSelect"
	
	def setUp(self):
		print(self.msg, self.device, self.dtype)
		self.volume = torch.rand(	self.batch_size, self.num_features, self.box_size_bin, self.box_size_bin, self.box_size_bin, 
									device=self.device, dtype=self.dtype, requires_grad=True)
		self.coords = torch.rand(self.batch_size, self.max_num_atoms*3, device=self.device, dtype=self.dtype)*self.box_size_ang
		self.num_atoms = torch.zeros(self.batch_size, dtype=torch.int, device=self.device).fill_(self.max_num_atoms)


class TestCoordsSelectForward(TestCoordsSelect):
	msg = "Testing CoordsSelect forward"
	def runTest(self):
		sv = CoordsSelect(box_size_bins=self.box_size_bin, box_size_ang=self.box_size_ang)
		features = sv(self.volume, self.coords, self.num_atoms)
				
		for batch_idx in range(self.batch_size):
			for feature_idx in range(self.num_features):
				res = float(self.box_size_bin)/float(self.box_size_ang)
				err = 0.0
				for i in range(0, self.num_atoms[batch_idx].item()):
					x = int(np.floor(self.coords[batch_idx, 3*i].item()*res))
					y = int(np.floor(self.coords[batch_idx, 3*i+1].item()*res))
					z = int(np.floor(self.coords[batch_idx, 3*i+2].item()*res))
					err += np.abs(features[batch_idx, feature_idx, i].item() - self.volume[batch_idx, feature_idx, x, y, z].item())
				
				self.assertAlmostEqual(err, 0.0)

class TestCoordsSelectBackward(TestCoordsSelect):
	msg = "Testing CoordsSelect backward"
	def runTest(self):
		sv = CoordsSelect(box_size_bins=self.box_size_bin, box_size_ang=self.box_size_ang)
		features = sv(self.volume, self.coords, self.num_atoms)
		features.backward(torch.ones_like(features))
		
		for batch_idx in range(self.batch_size):
			for feature_idx in range(self.num_features):
				res = float(self.box_size_bin)/float(self.box_size_ang)
				err = 0.0
				for i in range(0, self.num_atoms[batch_idx].item()):
					x = int(np.floor(self.coords[batch_idx, 3*i].item()*res))
					y = int(np.floor(self.coords[batch_idx, 3*i+1].item()*res))
					z = int(np.floor(self.coords[batch_idx, 3*i+2].item()*res))
					err += np.abs(1.0 - self.volume.grad[batch_idx, feature_idx, x, y, z].item())
				
				self.assertAlmostEqual(err, 0.0)



if __name__=='__main__':
	unittest.main()
