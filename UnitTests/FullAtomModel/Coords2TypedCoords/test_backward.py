import sys
import os
import torch
import numpy as np
import unittest
from TorchProteinLibrary import FullAtomModel

class TestCoords2TypedCoordsBackward(unittest.TestCase):
	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()
		self.c2tc = FullAtomModel.Coords2TypedCoords()
	
	def runTest(self):
		sequence = ['GGMLGWAHFGY']
		x0 = torch.zeros(1,7,len(sequence[0]), dtype=torch.double, device='cpu').requires_grad_()
		x1 = torch.zeros(1,7,len(sequence[0]), dtype=torch.double, device='cpu')
		x0.data[0,0,:] = -1.047
		x0.data[0,1,:] = -0.698
		x0.data[0,2:,:] = 110.4*np.pi/180.0

		y0, res, at, nat = self.a2c(x0, sequence)
		coords, num_atoms_of_type, offsets = self.c2tc(y0, res, at, nat)
		z0 = coords.sum()
			
		z0.backward()
		back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)
		
		error = 0.0
		N = 0
		for i in range(0,7):
			for j in range(0,x0.size(2)):
				dx = 0.0001
				x1.data.copy_(x0.data)
				x1.data[0,i,j]+=dx
				y1, res, at, nat = self.a2c(x1, sequence)
				coords, num_atoms_of_type, offsets = self.c2tc(y1, res, at, nat)
				z1 = coords.sum()
				dy_dx = (z1.data-z0.data)/(dx)
				error += torch.abs(dy_dx - back_grad_x0[0,i,j])
				N+=1
		error /= float(N)
		print('Error = ', error)
		self.assertLess(error, 0.01)

if __name__=='__main__':
	unittest.main()