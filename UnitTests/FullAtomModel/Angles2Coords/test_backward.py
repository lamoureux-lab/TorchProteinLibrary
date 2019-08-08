import sys
import os
import torch
import unittest
from TorchProteinLibrary import FullAtomModel
import numpy as np

class TestAngles2CoordsBackward(unittest.TestCase):
	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()

	def runTest(self):
		sequences = ['AFFAAGH', 'GGMLGWAHFGY', 'ACDEFGHIKLMNPQRSTVWY']	
		x0 = torch.zeros(len(sequences), 8, len(sequences[-1]), dtype=torch.double, device='cpu').requires_grad_()
		x1 = torch.zeros(len(sequences), 8, len(sequences[-1]), dtype=torch.double, device='cpu')
		x0.data[:,0,:] = -1.047
		x0.data[:,1,:] = -0.698
		x0.data[:,2,:] = np.pi
		x0.data[:,3:,:] = 110.4*np.pi/180.0
		
		y0, res, at, n_at = self.a2c(x0, sequences)
		y0 = y0.sum()
			
		y0.backward()
		back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)
		
		error = 0.0
		N = 0
		for b in range(0,len(sequences)):
			for i in range(0,8):
				for j in range(0,x0.size(2)):
					dx = 0.0001
					x1.data.copy_(x0.data)
					x1.data[b,i,j]+=dx
					y1, res, at, n_at = self.a2c(x1,sequences)
					y1 = y1.sum()
					dy_dx = (y1.data-y0.data)/(dx)
					error += np.abs(dy_dx - back_grad_x0.data[b,i,j])
					N+=1
		error/=float(N)
		print('Error = ', error)
		self.assertLess(error, 0.01)

if __name__=='__main__':
	unittest.main()