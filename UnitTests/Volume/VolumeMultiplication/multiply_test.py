import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
import random
import math
from TorchProteinLibrary.Volume import VolumeCrossMultiply
import _Volume

class TestVolumeCrossMultiplication(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 2
	num_features = 3
	eps=1e-03 
	atol=1e-05 
	rtol=0.001
	msg = "Testing VolumeCrossMultiplication"

	def setUp(self):
		print(self.msg, self.device, self.dtype)
		self.vc = VolumeCrossMultiply()
		self.box_size = 30
		self.resolution = 1.0

		self.vol1 = torch.zeros(self.batch_size, self.num_features, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		self.vol2 = torch.zeros(self.batch_size, self.num_features, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		self.T = torch.zeros(self.batch_size, 3)

		for i in range(self.batch_size):
			self.T[i, 0] = 5
			self.T[i, 1] = 5
			self.T[i, 2] = 5
			for j in range(self.num_features):
				self.vol1[i, j, 10, 10, 10] = 1.0*((-1)**j)
				self.vol2[i, j, 5, 5, 5] = 2.0*((-1)**(j+1))

	

class TestVolumeCrossMultiplicationForward(TestVolumeCrossMultiplication):
	msg = "Testing VolumeCrossMultiplicationFwd"
	def runTest(self):
		
		mult = self.vc(self.vol1, self.vol2, self.T)
		for batch_id in range(self.batch_size):
			l = 0
			for diag in range(int(self.num_features/2 + 1)):
				for k in range(self.num_features - diag):
					i = k
					j = diag + k
					self.assertEqual(
							mult[batch_id, l].item(), 
							self.vol1[batch_id, i, 10, 10, 10].item() * self.vol2[batch_id, j, 5, 5, 5].item()
							)
					l+=1
		

if __name__=='__main__':
	unittest.main()