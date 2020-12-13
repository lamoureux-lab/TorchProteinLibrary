import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
import random
import math
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords, Angles2Coords
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, CoordsRotate, getBBox
from TorchProteinLibrary.Volume import TypedCoords2Volume

from TorchProteinLibrary.Utils import ProteinBatch, ScalarField

class TestTypedCoords2Volume(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 2
	max_num_atoms = 32
	eps=1e-02 
	atol=1e-03 
	rtol=0.001
	msg = "Testing TypedCoords2Volume"

	def generatePotential(self, box_size):
		potential = torch.zeros(1, 11, box_size,box_size,box_size, dtype=self.dtype, device=self.device)
		for i in range(0,box_size):
			potential[0,:,i,:,:] = float(i)/float(box_size) - 0.5
		
		for i in range(0, 11):
			potential[0,i,:,:,:] *= random.random()

		return potential

	def generateAtoms(self, num_atoms, box_size):
		num_atoms_of_type = torch.zeros(1, 11, dtype=torch.int, device=self.device)
		coords = torch.zeros(1, 11, 3*self.max_num_atoms, dtype=self.dtype, device=self.device)
		
		for i in range(0, num_atoms):
			atom_type = np.random.randint(low=0, high=11)
			
			atom_index = num_atoms_of_type[0, atom_type].item()

			coords[0, atom_type, 3*atom_index + 0 ] = float(random.randrange(5.0, box_size-5.0))
			coords[0, atom_type, 3*atom_index + 1 ] = float(random.randrange(5.0, box_size-5.0))
			coords[0, atom_type, 3*atom_index + 2 ] = float(random.randrange(5.0, box_size-5.0))

			num_atoms_of_type[0, atom_type] += 1
		
		return coords, num_atoms_of_type
		

	def setUp(self):
		print(self.msg, self.device, self.dtype)
		
		self.box_size = 30
		self.resolution = 1.0
		self.tc2v = TypedCoords2Volume(box_size=self.box_size, resolution=self.resolution)

		b_coords = []
		b_num_atoms_of_type = []
		b_potential = []
		for i in range(self.batch_size):
			num_atoms = math.floor(0.5 * (1.0 + random.random()) * self.max_num_atoms)
			coords, num_atoms_of_type = self.generateAtoms(num_atoms, self.box_size)
			potential = self.generatePotential(self.box_size)
			b_coords.append(coords)
			b_num_atoms_of_type.append(num_atoms_of_type)
			b_potential.append(potential)

		self.coords = torch.cat(b_coords, dim=0)
		self.num_atoms_of_type = torch.cat(b_num_atoms_of_type, dim=0)
		self.potential = torch.cat(b_potential, dim=0)




class TestTypedCoords2VolumeForward(TestTypedCoords2Volume):
	def runTest(self):
		volume_gpu = self.tc2v(self.coords, self.num_atoms_of_type)
		volume = volume_gpu.to(device='cpu', dtype=torch.float)
		print(torch.max(volume[0,0,:,:,:]), torch.min(volume[0,0,:,:,:]), self.num_atoms_of_type[0,0])
		import matplotlib.pylab as plt
		fig = plt.figure(figsize=(10, 10))
		axis = fig.add_subplot(111, projection='3d')
		ScalarField(volume[0,0,:,:,:], resolution=self.resolution).isosurface(0.1, axis=axis, alpha=0.0)
		num_atoms = self.num_atoms_of_type[0,0].item()
		c = self.coords[0,0,:].view(int(self.coords.size(2)/3), 3)[:num_atoms, :]
		axis.scatter(c[:,0].cpu().numpy(), c[:,1].cpu().numpy(), c[:,2].cpu().numpy())
		plt.show()
		
		if not os.path.exists('TestFig'):
			os.mkdir('TestFig')

		for i in range(self.batch_size):
			for j in range(11):
				for k in range(self.num_atoms_of_type[i,j]):
					x = self.coords[i,j,3*k+0].item()
					y = self.coords[i,j,3*k+1].item()
					z = self.coords[i,j,3*k+2].item()
					x_i = int(x/self.resolution)
					y_i = int(y/self.resolution)
					z_i = int(z/self.resolution)
					xc = x_i * self.resolution
					yc = y_i * self.resolution
					zc = z_i * self.resolution
					r2 = (x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc)
					self.assertGreaterEqual(volume_gpu[i,j,x_i,y_i,z_i].item(), np.exp(-r2/2.0))
		
		#_Volume.Volume2Xplor(volume[0,:,:,:], "TestFig/total_b0_vtest_%d_%.1f.xplor"%(self.box_size, self.resolution), self.resolution)

class TestTypedCoords2VolumeForward_Double(TestTypedCoords2VolumeForward):
	dtype=torch.double


class TestTypedCoords2VolumeBackward(TestTypedCoords2Volume):
	msg = "Testing TypedCoords2VolumeBackward"
	def runTest(self):
		coords0 = torch.zeros_like(self.coords).copy_(self.coords).requires_grad_()
		volume_gpu = self.tc2v(coords0, self.num_atoms_of_type)
		E0 = torch.sum(volume_gpu*self.potential)
		E0.backward()
		eps = 0.01
		for i in range(0, self.coords.size(0)):
			for j in range(0, self.coords.size(1)):
				for k in range(0, self.coords.size(2)):
					coords1 = torch.zeros_like(self.coords).copy_(self.coords)
					coords1[i,j,k] += eps
					volume_gpu1 = self.tc2v(coords1, self.num_atoms_of_type)
					E1 = torch.sum(volume_gpu1*self.potential)
					dE_dx = (E1.item() - E0.item())/(eps)
					print(dE_dx, coords0.grad[i,j,k].item())
					self.assertLess(math.fabs(dE_dx - coords0.grad[i,j,k].item()), math.fabs(coords0.grad[i,j,k].item()) * self.rtol + self.atol)

class TestTypedCoords2VolumeBackward_Double(TestTypedCoords2VolumeBackward):
	dtype=torch.double

if __name__=='__main__':
	unittest.main()