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

from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather

class TestTypedCoords2Volume(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 2
	max_num_atoms = 16
	eps=1e-03 
	atol=1e-05 
	rtol=0.001
	msg = "Testing TypedCoords2Volume"

	def generateAtoms(self, num_atoms, box_size):
		num_atoms_of_type = torch.zeros(1, 11, dtype=torch.int, device=self.device)
		coords = torch.zeros(1, 11, 3*self.max_num_atoms, dtype=self.dtype, device=self.device)
		
		for i in range(0, num_atoms):
			atom_type = np.random.randint(low=0, high=11)
			
			atom_index = num_atoms_of_type[0, atom_type].item()

			coords[0, atom_type, 3*atom_index + 0 ] = float(random.randrange(10.0, box_size-10.0))
			coords[0, atom_type, 3*atom_index + 1 ] = float(random.randrange(10.0, box_size-10.0))
			coords[0, atom_type, 3*atom_index + 2 ] = float(random.randrange(10.0, box_size-10.0))

			num_atoms_of_type[0, atom_type] += 1
		
		return coords, num_atoms_of_type
		
	def setUp(self):
		print(self.msg, self.device, self.dtype)
		
		self.box_size = 30
		self.resolution = 1.0
		self.tc2v = TypedCoords2Volume(box_size=self.box_size, resolution=self.resolution)
		self.tc2v_parallel = torch.nn.DataParallel(self.tc2v, device_ids=[0,1])

		b_coords = []
		b_num_atoms_of_type = []
		for i in range(self.batch_size):
			num_atoms = math.floor(0.5 * (1.0 + random.random()) * self.max_num_atoms)
			coords, num_atoms_of_type = self.generateAtoms(num_atoms, self.box_size)
			b_coords.append(coords)
			b_num_atoms_of_type.append(num_atoms_of_type)

		self.coords = torch.cat(b_coords, dim=0)
		self.num_atoms_of_type = torch.cat(b_num_atoms_of_type, dim=0)

class TestTypedCoords2VolumeForward(TestTypedCoords2Volume):
	def runTest(self):
			  
		# kwargs = {}
		# inputs, kwargs = scatter_kwargs((type_coords, num_atoms_of_type), kwargs, self.device_ids, dim=0)
		# print(inputs)
		# replicas = replicate(self.project, self.device_ids[:len(inputs)], not torch.is_grad_enabled())
		# print(replicas)
		# outputs = parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
		# volume_gpu = self.tc2v(self.coords, self.num_atoms_of_type)

		volume_multi = self.tc2v_parallel(self.coords, self.num_atoms_of_type)
		volume_single = self.tc2v(self.coords, self.num_atoms_of_type)
		print(volume_multi.size(), volume_single.size())
		print(volume_multi.sum(), volume_single.sum())
		print(torch.abs(volume_multi-volume_single).sum())
		self.assertLess(torch.max(torch.abs(volume_multi-volume_single)).item(), self.atol + self.rtol * torch.max(torch.abs(volume_single)).item())
		
if __name__=='__main__':
	unittest.main()