import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
import random
import math
from TorchProteinLibrary.Volume import VolumeCrossConvolution, VolumeCrossMultiply

class TestVolumeCrossConvolutionVSMultiplication(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 2
	num_features = 3
	box_size = 30
	resolution = 1.0
	eps=1e-03 
	atol=1e-05 
	rtol=0.001
	msg = "Testing VolumeCrossConvolution versus CrossMultiplication"

	def setUp(self):
		print(self.msg, self.device, self.dtype)
		self.vc = VolumeCrossConvolution()
		self.mult = VolumeCrossMultiply()
		

		self.vol1 = torch.zeros(self.batch_size, self.num_features, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		self.vol2 = torch.zeros(self.batch_size, self.num_features, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		self.T = torch.zeros(self.batch_size, 3, dtype=self.dtype, device=self.device)
		self.R = torch.zeros(self.batch_size, 3, dtype=self.dtype, device=self.device)

		for i in range(self.batch_size):
			self.T[i, 0] = 5
			self.T[i, 1] = 5
			self.T[i, 2] = 5
			for j in range(self.num_features):
				self.vol1[i, j, 10, 10, 10] = 1.0*((-1)**j)
				self.vol2[i, j, 5, 5, 5] = 2.0*((-1)**(j+1))

	def runTest(self):
		mult = self.mult(self.vol1, self.vol2, self.R, self.T)
		conv = self.vc(self.vol1, self.vol2)
		for i in range(self.batch_size):
			for j in range(conv.size(1)):
				a = conv[i,j,self.box_size + 5, self.box_size + 5, self.box_size + 5].item()
				b = mult[i,j]
				self.assertLess(math.fabs(a - b), math.fabs(a) * self.rtol + self.atol)


if __name__=='__main__':
	unittest.main()