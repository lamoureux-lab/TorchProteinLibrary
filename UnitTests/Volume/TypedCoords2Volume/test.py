import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords, Angles2Coords
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, CoordsRotate, getBBox
from TorchProteinLibrary.Volume import TypedCoords2Volume
import _Volume

class TestTypedCoords2Volume(unittest.TestCase):
	def setUp(self):
		self.box_size = 10
		self.resolution = 1.0
		self.tc2v = TypedCoords2Volume(box_size=self.box_size, resolution=self.resolution, mode='gauss')
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.device = 'cuda'
		self.num_atoms_types = 11

		sequence = ['GGG', 'AAA']
		angles = torch.zeros(len(sequence), 7,len(sequence[0]), dtype=torch.double, device='cpu')
		angles[0,0,:] = -1.047
		angles[0,1,:] = -0.698
		angles[0,2:,:] = np.pi
		angles[0,3:,:] = 110.4*np.pi/180.0
		a2c = Angles2Coords()
		c2tc = Coords2TypedCoords()
		coords, res_names, atom_names, num_atoms = a2c(angles, sequence)
		tcoords, self.num_atoms_of_type, self.offsets = c2tc(coords, res_names, atom_names, num_atoms)
		a,b = getBBox(tcoords, num_atoms)
		center = a+b/2.0 - (self.box_size*self.resolution)/2.0
		self.ccoords = self.translate(tcoords, -center, num_atoms)


class TestTypedCoords2VolumeForward(TestTypedCoords2Volume):
	def runTest(self):
		volume_gpu = self.tc2v(self.ccoords.to(device='cuda'), self.num_atoms_of_type.to(device='cuda'), self.offsets.to(device='cuda'))
		volume = volume_gpu.sum(dim=1).squeeze().cpu()
		
		if not os.path.exists('TestFig'):
			os.mkdir('TestFig')
		_Volume.Volume2Xplor(volume[0,:,:,:], "TestFig/total_b0_vtest_%d_%.1f.xplor"%(self.box_size, self.resolution), self.resolution)
		_Volume.Volume2Xplor(volume[1,:,:,:], "TestFig/total_b1_vtest_%d_%.1f.xplor"%(self.box_size, self.resolution), self.resolution)


# class TestTypedCoords2VolumeBackward(TestTypedCoords2Volume):
# 	def runTest(self):
# 		grad_coords = self.ccoords.to(device='cuda').requires_grad_()
# 		naot = self.num_atoms_of_type.to(device='cuda')
# 		off = self.offsets.to(device='cuda')
# 		result = torch.autograd.gradcheck(self.tc2v, (grad_coords, naot, off), eps=1e-3, atol=1e-2, rtol=0.01)
# 		self.assertTrue(result)

class TestTypedCoords2VolumeBackwardSA(TestTypedCoords2Volume):
	def runTest(self):
		coords = torch.zeros(1, 3, device='cuda', dtype=torch.double).fill_(self.box_size*self.resolution/2.0).requires_grad_()
		offsets = torch.zeros(1, device='cuda', dtype=torch.int)
		num_atoms_of_type = torch.ones(1, device='cuda', dtype=torch.int)
		result = torch.autograd.gradcheck(self.tc2v, (coords, num_atoms_of_type, offsets), eps=1e-3, atol=1e-2, rtol=0.01)
		self.assertTrue(result)

if __name__=='__main__':
	unittest.main()