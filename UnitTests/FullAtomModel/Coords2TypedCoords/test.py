import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords

class TestCoords2TypedCoords(unittest.TestCase):

	def setUp(self):
		self.sequence = ['GGGGGG', 'GGGGGG']
		angles = torch.zeros(len(self.sequence), 7,len(self.sequence[0]), dtype=torch.double, device='cpu')
		angles[0,0,:] = -1.047
		angles[0,1,:] = -0.698
		angles[0,2:,:] = np.pi
		angles[0,3:,:] = 110.4*np.pi/180.0
		a2c = Angles2Coords()
		self.coords, self.res_names, self.atom_names, self.num_atoms = a2c(angles, self.sequence)
		self.c2tc = Coords2TypedCoords()

class TestCoords2TypedCoordsForward(TestCoords2TypedCoords):
	def runTest(self):		
		tcoords, num_atoms_of_type, offsets = self.c2tc(self.coords, self.res_names, self.atom_names, self.num_atoms)
		self.assertEqual(num_atoms_of_type[0,0].item(), 0) #sulfur 
		self.assertEqual(num_atoms_of_type[0,1].item(), len(self.sequence[0])) #nitrogen amide
		self.assertEqual(num_atoms_of_type[0,2].item(), 0) #nitrogen arom
		self.assertEqual(num_atoms_of_type[0,3].item(), 0) #nitrogen guan
		self.assertEqual(num_atoms_of_type[0,4].item(), 0) #nitrogen ammon
		self.assertEqual(num_atoms_of_type[0,5].item(), len(self.sequence[0])) #oxygen carbonyl
		self.assertEqual(num_atoms_of_type[0,6].item(), 0) #oxygen hydroxyl
		self.assertEqual(num_atoms_of_type[0,7].item(), 0) #oxygen carboxyl
		self.assertEqual(num_atoms_of_type[0,8].item(), len(self.sequence[0])) #oxygen carboxyl
		self.assertEqual(num_atoms_of_type[0,9].item(), 0) #carbon sp2 
		self.assertEqual(num_atoms_of_type[0,10].item(), len(self.sequence[0])) #carbon sp3

		for i in range(1, len(self.sequence)): #batch check
			for j in range(11):
				self.assertEqual(num_atoms_of_type[i,j], num_atoms_of_type[i-1,j])

class TestCoords2TypedCoordsBackward(TestCoords2TypedCoords):
	def runTest(self):
		self.coords.requires_grad_()
		tcoords, num_atoms_of_type, offsets = self.c2tc(self.coords, self.res_names, self.atom_names, self.num_atoms)
		z0 = tcoords.sum()	
		z0.backward()
		back_grad_x0 = torch.zeros_like(self.coords).copy_(self.coords.grad)
		
		error = 0.0
		N = 0
		x1 = torch.zeros_like(self.coords)
		for i in range(0,len(self.sequence)):
			for j in range(0,tcoords.size(1)):
				dx = 0.0001
				x1.copy_(self.coords)
				x1[i,j] += dx
				x1coords, num_atoms_of_type, offsets = self.c2tc(x1, self.res_names, self.atom_names, self.num_atoms)
				z1 = x1coords.sum()
				dy_dx = (z1-z0)/(dx)
				error += torch.abs(dy_dx - back_grad_x0[i,j]).item()
				N+=1

		error /= float(N)
		self.assertLess(error, 1E-5)


if __name__=="__main__":
	unittest.main()
