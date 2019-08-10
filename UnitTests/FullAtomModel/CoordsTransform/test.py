import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, getBBox, CoordsRotate, getRandomRotation

class TestCoordsTransform(unittest.TestCase):
	def _plot_coords(coords, filename):
		if not os.path.exists("TestFig"):
			os.mkdir("TestFig")

		min_xyz = -1.5
		max_xyz = 1.5
		coords = coords.numpy()
		sx, sy, sz = coords[:,0], coords[:,1], coords[:,2]
		fig = plt.figure()
		ax = p3.Axes3D(fig)
		ax.plot(sx, sy, sz, '.', label = plot_name)
		ax.set_xlim(min_xyz,max_xyz)
		ax.set_ylim(min_xyz,max_xyz)
		ax.set_zlim(min_xyz,max_xyz)
		ax.legend()
		plt.savefig('TestFig/%s'%filename)

	def setUp(self):
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.num_atoms = torch.ones(1, dtype=torch.int, device='cpu')
		self.coords = torch.zeros(1,3, dtype=torch.double, device='cpu')
		self.coords[0,0] = 1.0
		self.coords[0,1] = 0.0
		self.coords[0,2] = 0.0
		

class TestCoordsTranslateForward(TestCoordsTransform):
	def runTest(self):
		a,b = getBBox(self.coords, self.num_atoms)
		center = (a+b)*0.5
		centered_coords = self.translate(self.coords, -center, self.num_atoms)
		a,b = getBBox(centered_coords, self.num_atoms)
		center = (a+b)*0.5
		self.assertAlmostEqual(center[0,0].item(), 0.0)
		self.assertAlmostEqual(center[0,1].item(), 0.0)
		self.assertAlmostEqual(center[0,2].item(), 0.0)

class TestCoordsRotateForward(TestCoordsTransform):
	def runTest(self):
		#CCW rotation around y by 90deg
		U = torch.tensor([[	[0, 0, 1],
							[0, 1, 0],
							[-1, 0, 0]]], dtype=torch.double, device='cpu')
		rot_coords = self.rotate(self.coords, U, self.num_atoms)
		self.assertAlmostEqual(rot_coords[0,0].item(), 0.0)
		self.assertAlmostEqual(rot_coords[0,1].item(), 0.0)
		self.assertAlmostEqual(rot_coords[0,2].item(), -1.0)

class TestRandomRotations(TestCoordsTransform):
	def runTest(self):
		batch_size = 30
		R = getRandomRotation(batch_size)
		#Matrixes should be det = 1
		for i in range(R.size(0)):
			U = R[i].numpy()
			D = np.linalg.det(U)
			self.assertAlmostEqual(D, 1.0)

class TestBBox(TestCoordsTransform):
	def runTest(self):
		coords = torch.tensor([[	1.0, -1.0, -2.0, #r0
									2.0, 0.0, 0.0, #r1
									0.0, 1.0, 1.0 #r3
								]])
		num_atoms = torch.tensor([3], dtype=torch.int)
		a,b = getBBox(coords, num_atoms)
		self.assertAlmostEqual(a[0,0].item(), 0.0)
		self.assertAlmostEqual(a[0,1].item(), -1.0)
		self.assertAlmostEqual(a[0,2].item(), -2.0)
		
		self.assertAlmostEqual(b[0,0].item(), 2.0)
		self.assertAlmostEqual(b[0,1].item(), 1.0)
		self.assertAlmostEqual(b[0,2].item(), 1.0)
	


if __name__ == '__main__':
	unittest.main()

