import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, getBBox, CoordsRotate, getRandomRotation, Coords2Center

class TestCoordsTransform(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 7
	batch_size = 16
	max_num_atoms = 30
	eps=1e-06 
	atol=1e-05 
	rtol=0.001
	msg = "Testing transforms"
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
		print(self.msg, self.device, self.dtype)
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()
		self.getCenter = Coords2Center()
		self.coords = torch.randn(self.batch_size, self.max_num_atoms*3, dtype=self.dtype, device=self.device)
		self.num_atoms = torch.zeros(self.batch_size, dtype=torch.int, device=self.device).random_(int(self.max_num_atoms/2), self.max_num_atoms)
		


class TestCoords2CenterForward(TestCoordsTransform):
	msg = "Testing Coords2Center fwd:"
	def runTest(self):
		center = self.getCenter(self.coords, self.num_atoms)

		coords_shaped = self.coords.reshape(self.batch_size, self.max_num_atoms, 3)
	
		for i in range(self.batch_size):
			man_center = coords_shaped[i, :self.num_atoms[i].item(), :].sum(dim=0)/float(self.num_atoms[i].item())
			self.assertAlmostEqual(man_center[0].item(), center[i,0].item(), places=self.places)
			self.assertAlmostEqual(man_center[1].item(), center[i,1].item(), places=self.places)
			self.assertAlmostEqual(man_center[2].item(), center[i,2].item(), places=self.places)

class TestCoords2CenterForward_CPUFloat(TestCoords2CenterForward):
	device = 'cpu'
	dtype = torch.float
	places = 5

class TestCoords2CenterForward_CUDADouble(TestCoords2CenterForward):
	device = 'cuda'
	dtype = torch.double

class TestCoords2CenterForward_CUDAFloat(TestCoords2CenterForward):
	device = 'cuda'
	dtype = torch.float
	places = 5
		

class TestCoords2CenterBackward(TestCoordsTransform):
	msg = "Testing Coords2Center bwd:"
	def runTest(self):
		coords = torch.zeros_like(self.coords).copy_(self.coords).requires_grad_()
		result = torch.autograd.gradcheck(self.getCenter, (coords, self.num_atoms), self.eps, self.atol, self.rtol)
		self.assertTrue(result)

class TestCoords2CenterBackward_CPUFloat(TestCoords2CenterBackward):
	device = 'cpu'
	dtype = torch.float
	places = 5
	eps=1e-03 
	atol=1e-04 
	rtol=0.01

class TestCoords2CenterBackward_CUDADouble(TestCoords2CenterBackward):
	device = 'cuda'
	dtype = torch.double

class TestCoords2CenterBackward_CUDAFloat(TestCoords2CenterBackward):
	device = 'cuda'
	dtype = torch.float
	places = 5
	eps=1e-03 
	atol=1e-04 
	rtol=0.01


class TestCoordsTranslateForward(TestCoordsTransform):
	msg = "Testing CoordsTranslate fwd:"
	def runTest(self):
		center = self.getCenter(self.coords, self.num_atoms)
		centered_coords = self.translate(self.coords, -center, self.num_atoms)
		center_centered = self.getCenter(centered_coords, self.num_atoms)
		for i in range(self.batch_size):
			self.assertAlmostEqual(center_centered[i,0].item(), 0.0, places=self.places)
			self.assertAlmostEqual(center_centered[i,1].item(), 0.0, places=self.places)
			self.assertAlmostEqual(center_centered[i,2].item(), 0.0, places=self.places)

class TestCoordsTranslateForwardCPUFloat(TestCoordsTranslateForward):
	device = 'cpu'
	dtype = torch.float
	places = 5

class TestCoordsTranslateForwardCUDADouble(TestCoordsTranslateForward):
	device = 'cuda'
	dtype = torch.double

class TestCoordsTranslateForwardCUDAFloat(TestCoordsTranslateForward):
	device = 'cuda'
	dtype = torch.float
	places = 5

class TestCoordsTranslateBackward(TestCoordsTransform):
	msg = "Testing CoordsTranslate bwd:"
	def runTest(self):
		coords = torch.zeros_like(self.coords).copy_(self.coords).requires_grad_()
		T = torch.randn(self.batch_size, 3, dtype=self.dtype, device=self.device)
		result = torch.autograd.gradcheck(self.translate, (coords, T, self.num_atoms), self.eps, self.atol, self.rtol)
		self.assertTrue(result)



class TestCoordsRotateForward(TestCoordsTransform):
	msg = "Testing CoordsRotate fwd:"
	def runTest(self):

		#CCW rotation around y by 90deg
		R = torch.cat([torch.tensor([[	[0, 0, 1],
										[0, 1, 0],
										[-1, 0, 0]]], 
										dtype=self.dtype, device=self.device) for i in range(self.batch_size)], dim=0)

		rot_coords = self.rotate(self.coords, R, self.num_atoms)
		for i in range(self.batch_size):
			for j in range(self.num_atoms[i].item()):
				self.assertAlmostEqual(rot_coords[i, 3*j + 0].item(), self.coords[i, 3*j + 2].item(), places=self.places)
				self.assertAlmostEqual(rot_coords[i, 3*j + 1].item(), self.coords[i, 3*j + 1].item(), places=self.places)
				self.assertAlmostEqual(rot_coords[i, 3*j + 2].item(), -self.coords[i, 3*j + 0].item(), places=self.places)

class TestCoordsRotateForward_CPUFloat(TestCoordsRotateForward):
	device = 'cpu'
	dtype = torch.float
	places = 5

class TestCoordsRotateForward_CUDADouble(TestCoordsRotateForward):
	device = 'cuda'
	dtype = torch.double

class TestCoordsRotateForward_CUDAFloat(TestCoordsRotateForward):
	device = 'cuda'
	dtype = torch.float
	places = 5

class TestCoordsRotateBackward(TestCoordsTransform):
	msg = "Testing CoordsRotate bwd:"
	def runTest(self):
		coords = torch.zeros_like(self.coords).copy_(self.coords).requires_grad_()
		
		#Random rotation matrixes
		R = getRandomRotation(self.batch_size).to(dtype=self.dtype, device=self.device)
		result = torch.autograd.gradcheck(self.rotate, (coords, R, self.num_atoms))
		self.assertTrue(result)

class TestCoordsRotateBackward_CPUFloat(TestCoordsRotateBackward):
	device = 'cpu'
	dtype = torch.float
	places = 5
	eps=1e-05 
	atol=1e-03 
	rtol=0.01

class TestCoordsRotateBackward_CUDADouble(TestCoordsRotateBackward):
	device = 'cuda'
	dtype = torch.double

class TestCoordsRotateBackward_CUDAFloat(TestCoordsRotateBackward):
	device = 'cuda'
	dtype = torch.float
	places = 5
	eps=1e-05 
	atol=1e-03 
	rtol=0.01
		
		
class TestRandomRotations(TestCoordsTransform):
	msg = "Testing RandomRotations:"
	def runTest(self):
		R = getRandomRotation(self.batch_size)
		#Matrixes should be det = 1
		for i in range(R.size(0)):
			U = R[i].numpy()
			D = np.linalg.det(U)
			self.assertAlmostEqual(D, 1.0, places=self.places)

class TestRandomRotations_CPUFloat(TestRandomRotations):
	device = 'cpu'
	dtype = torch.float
	places = 5

class TestBBox(TestCoordsTransform):
	msg = "Testing BBox:"
	def runTest(self):
		coords = torch.tensor([[	1.0, -1.0, -2.0, #r0
									2.0, 0.0, 0.0, #r1
									0.0, 1.0, 1.0 #r3
								]])
		num_atoms = torch.tensor([3], dtype=torch.int)
		a,b = getBBox(coords, num_atoms)
		self.assertAlmostEqual(a[0,0].item(), 0.0, places=self.places)
		self.assertAlmostEqual(a[0,1].item(), -1.0, places=self.places)
		self.assertAlmostEqual(a[0,2].item(), -2.0, places=self.places)
		
		self.assertAlmostEqual(b[0,0].item(), 2.0, places=self.places)
		self.assertAlmostEqual(b[0,1].item(), 1.0, places=self.places)
		self.assertAlmostEqual(b[0,2].item(), 1.0, places=self.places)
	
class TestBBox_CPUFloat(TestBBox):
	device = 'cpu'
	dtype = torch.float
	places = 5

if __name__ == '__main__':
	unittest.main()

