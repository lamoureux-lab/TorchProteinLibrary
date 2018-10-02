import sys
import os
import torch
import unittest
import numpy as np
from TorchProteinLibrary import FullAtomModel

class TestCoords2TypedCoordsBackward(unittest.TestCase):
	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()
		self.c2tc = FullAtomModel.Coords2TypedCoords()
		self.c2cc = FullAtomModel.CoordsTransform.Coords2CenteredCoords(rotate=True, translate=True)
		self.error = 0.0
		self.N = 0
			
	def runTest(self):
		sequence = ['GGMLGWAHFGY']
		x0 = torch.zeros(1, 7, len(sequence[0]), dtype=torch.double).requires_grad_()
		x1 = torch.zeros(1, 7, len(sequence[0]), dtype=torch.double)
		x0.data[0,0,:] = -1.047
		x0.data[0,1,:] = -0.698
		x0.data[0,2:,:] = 110.4*np.pi/180.0
				
		y0, res, at, num_atoms = self.a2c(x0, sequence)
		y0 = self.c2cc(y0, num_atoms)
		coords, num_atoms_of_type, offsets = self.c2tc(y0, res, at, num_atoms)
		coords = coords.resize(1, int(coords.size(1)/3), 3)
		center_mass = coords.mean(dim=1).unsqueeze(dim=1)
		coords = coords - center_mass
		Rg = torch.mean(torch.sqrt((coords*coords).sum(dim=2)))	
		Rg.backward()
		back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.double).copy_(x0.grad.data)

		x1.data.copy_(x0.data)
		for i in range(0,7):
			grads = []
			for j in range(0,x0.size(2)):
				dx = 0.0001
				x1.data.copy_(x0.data)
				x1.data[0,i,j]+=dx
				y1, res, at, num_atoms = self.a2c(x1, sequence)
				coords, num_atoms_of_type, offsets = self.c2tc(y1, res, at, num_atoms)
				coords = coords.resize(1, int(coords.size(1)/3), 3)
				center_mass = coords.mean(dim=1).unsqueeze(dim=1)
				coords = coords - center_mass
				Rg_1 = torch.mean(torch.sqrt((coords*coords).sum(dim=2)))	
				
				dy_dx = (Rg_1.data-Rg.data)/(dx)
				grads.append(dy_dx)
				self.error += torch.abs(dy_dx - back_grad_x0[0,i,j])
				self.N+=1

		self.error /= float(self.N)
		print('Error = ', self.error)
		self.assertLess(self.error, 0.01)


if __name__=='__main__':
	unittest.main()