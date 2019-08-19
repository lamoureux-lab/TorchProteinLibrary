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
import _Volume

class TestTypedCoords2Volume(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 4
	max_num_atoms = 16
	eps=1e-03 
	atol=1e-05 
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
		atom_coords = []
		atom_types = []
		for i in range(0, num_atoms):
			atom_coords.append(1.0 + np.random.rand(3)*box_size)
			atom_types.append(np.random.randint(low=0, high=11))
		
		num_atoms_of_type = torch.zeros(1,11, dtype=torch.int, device=self.device)
		offsets = torch.zeros(1,11, dtype=torch.int, device=self.device)
		coords = torch.zeros(1, 3*self.max_num_atoms, dtype=self.dtype, device=self.device)
		
		for atom_type in range(0,11):
			
			for i, atom in enumerate(atom_types):
				if atom == atom_type:
					num_atoms_of_type[0,atom_type]+=1
			
			if atom_type>0:
				offsets[0, atom_type] = offsets[0, atom_type-1] + num_atoms_of_type[0, atom_type-1]
		
		current_num_atoms_of_type = [0 for i in range(11)]
		for i, r in enumerate(atom_coords):
			index = 3*offsets[0, atom_types[i]] + 3*current_num_atoms_of_type[atom_types[i]]
			coords[0, index + 0 ] = r[0]
			coords[0, index + 1 ] = r[1]
			coords[0, index + 2 ] = r[2]
			current_num_atoms_of_type[atom_types[i]] += 1

		return coords, num_atoms_of_type, offsets
		

	def setUp(self):
		print(self.msg, self.device, self.dtype)
		
		self.box_size = 30
		self.resolution = 1.0
		self.tc2v = TypedCoords2Volume(box_size=self.box_size, resolution=self.resolution)

		b_coords = []
		b_num_atoms_of_type = []
		b_offsets = []
		b_potential = []
		for i in range(self.batch_size):
			num_atoms = math.floor(0.5 * (1.0 + random.random()) * self.max_num_atoms)
			coords, num_atoms_of_type, offsets = self.generateAtoms(num_atoms, self.box_size)
			potential = self.generatePotential(self.box_size)
			b_coords.append(coords)
			b_num_atoms_of_type.append(num_atoms_of_type)
			b_offsets.append(offsets)
			b_potential.append(potential)

		self.coords = torch.cat(b_coords, dim=0)
		self.num_atoms_of_type = torch.cat(b_num_atoms_of_type, dim=0)
		self.offsets = torch.cat(b_offsets, dim=0)
		self.potential = torch.cat(b_potential, dim=0)




class TestTypedCoords2VolumeForward(TestTypedCoords2Volume):
	def runTest(self):
		volume_gpu = self.tc2v(self.coords, self.num_atoms_of_type, self.offsets)
		volume = volume_gpu.sum(dim=1).to(device='cpu', dtype=torch.float)
		
		if not os.path.exists('TestFig'):
			os.mkdir('TestFig')
		_Volume.Volume2Xplor(volume[0,:,:,:], "TestFig/total_b0_vtest_%d_%.1f.xplor"%(self.box_size, self.resolution), self.resolution)

class TestTypedCoords2VolumeForward_Double(TestTypedCoords2VolumeForward):
	dtype=torch.double


class TestTypedCoords2VolumeBackward(TestTypedCoords2Volume):
	msg = "Testing TypedCoords2VolumeBackward"
	def runTest(self):
		coords0 = torch.zeros_like(self.coords).copy_(self.coords).requires_grad_()
		volume_gpu = self.tc2v(coords0, self.num_atoms_of_type, self.offsets)
		E0 = torch.sum(volume_gpu*self.potential)
		E0.backward()
		
		for i in range(0, self.coords.size(0)):
			for j in range(0, self.coords.size(1)):
				coords1 = torch.zeros_like(self.coords).copy_(self.coords)
				coords1[i,j] += self.eps
				volume_gpu1 = self.tc2v(coords1, self.num_atoms_of_type, self.offsets)
				E1 = torch.sum(volume_gpu1*self.potential)
				dE_dx = (E1.item() - E0.item())/(self.eps)
				self.assertLess(math.fabs(dE_dx - coords0.grad[i,j].item()), math.fabs(E0.item()) * self.rtol + self.atol)

class TestTypedCoords2VolumeBackward_Double(TestTypedCoords2VolumeBackward):
	dtype=torch.double

if __name__=='__main__':
	unittest.main()