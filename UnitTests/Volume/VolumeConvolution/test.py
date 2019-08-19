import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
import random
import math
from TorchProteinLibrary.Volume import VolumeConvolution
import _Volume

class TestVolumeConvolution(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	places = 5
	batch_size = 4
	max_num_atoms = 16
	eps=1e-03 
	atol=1e-05 
	rtol=0.001
	msg = "Testing VolumeConvolution"

	def setUp(self):
		print(self.msg, self.device, self.dtype)
		self.vc = VolumeConvolution()
		self.box_size = 30
		self.resolution = 1.0
		
	def fill_V1(self, r0, r1, R0, R1, volume):
		volume_size = volume.size(2)
		for x in range(volume_size):
			for y in range(volume_size):
				for z in range(volume_size):
					r_0 = np.array([x,y,z]) - r0
					r_1 = np.array([x,y,z]) - r1
					dist0 = np.linalg.norm(r_0)
					dist1 = np.linalg.norm(r_1)
					if dist0 < R0 and dist1 > R1:
						volume.data[0,0,x,y,z] = 1.0	

	def fill_V2(self, r0, R0, volume):
		volume_size = volume.size(2)
		for x in range(volume_size):
			for y in range(volume_size):
				for z in range(volume_size):
					r_0 = np.array([x,y,z]) - r0
					dist0 = np.linalg.norm(r_0)
					if dist0 < R0:
						volume.data[0,0,x,y,z] = 1.0	

	def get_boundary(self, volume_in, volume_out):
		volume_size = volume_in.size(2)
		for x in range(1,volume_size-1):
			for y in range(1,volume_size-1):
				for z in range(1,volume_size-1):
					cs = torch.sum(volume_in.data[0,0, x-1:x+2, y-1:y+2, z-1:z+2]).item()
					cc = volume_in.data[0,0,x,y,z].item()
					if (cc<0.01) and (cs>0.9):
						volume_out.data[0,0,x,y,z] = 1.0

	def get_argmax(self, volume):
		volume_size = volume.size(0)
		arg = (0,0,0)
		max_val = volume.data[0,0,0]
		
		for x in range(volume_size):
			for y in range(volume_size):
				for z in range(volume_size):
					val = float(volume.data[x,y,z])
					if val>max_val:
						arg = (x,y,z)
						max_val = val

		return max_val, arg



class TestVolumeConvolutionForward(TestVolumeConvolution):
	msg = "Testing VolumeConvolutionFwd"
	def runTest(self):
		R0 = 8
		x0 = [15,15,15]
		x1 = [23,15,15]
		
		y0 = [15,20,15]
		R1 = 5
		
		inp1 = torch.zeros(1, 1, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		inp1_border = torch.zeros_like(inp1)
		inp2 = torch.zeros_like(inp1)
		inp2_border = torch.zeros_like(inp1)
		
		self.fill_V1(np.array(x0), np.array(x1), R0, R1, inp1)
		self.get_boundary(inp1, inp1_border)
		
		self.fill_V2(np.array(y0), R1, inp2)
		# self.get_boundary(inp2, inp2_border)

		border_overlap = self.vc(inp1_border, inp2)
		bulk_overlap = self.vc(inp1, inp2)
		
		out = border_overlap - 0.5*bulk_overlap

		score, r = self.get_argmax(out[0,0,:,:,:])
		
		r = list(r)
		if r[0]>=self.box_size:
			r[0] = -(2*self.box_size - r[0])
		if r[1]>=self.box_size:
			r[1] = -(2*self.box_size - r[1])
		if r[2]>=self.box_size:
			r[2] = -(2*self.box_size - r[2])

		self.assertEqual(r[0]+y0[0], x1[0])
		self.assertEqual(r[1]+y0[1], x1[1])
		self.assertEqual(r[2]+y0[2], x1[2])

		# out2 = torch.zeros_like(inp1)
		# self.fill_V2(np.array([y0[0]+r[0],y0[1]+r[1],y0[2]+r[2]]), R1, out2)
		# v_out = inp1 + out2
		# _Volume.Volume2Xplor(v_out.squeeze().cpu(), "vout.xplor", 1.0)


class TestVolumeConvolutionBackward(TestVolumeConvolution):
	msg = "Testing VolumeConvolutionBwd"
	eps=4.0
	atol=1.0
	rtol=0.01
	def runTest(self):
		inp1 = torch.zeros(1, 1, self.box_size, self.box_size, self.box_size, dtype=self.dtype, device=self.device)
		inp2tmp_p = torch.zeros_like(inp1)
		inp2tmp_m = torch.zeros_like(inp1)
		inp2 = torch.zeros_like(inp1)
		inp1.requires_grad_()
		inp2.requires_grad_()

		target = torch.zeros(1, 1, 2*self.box_size, 2*self.box_size, 2*self.box_size, dtype=self.dtype, device=self.device)
		r0 = [20,15,15]
		r1 = [23,15,7]
		l0 = [7,24,7]
		self.fill_V2(np.array([12, 30, 50]), 5, target)
		self.fill_V1(np.array(r0), np.array(r1), 8, 5, inp1)
		self.fill_V2(np.array(l0), 5, inp2)

		out = self.vc(inp1, inp2)
		E0 = torch.sum((out-target)*(out-target))
		E0.backward()
		
		# num_grads = []
		# an_grads = []
		for i in range(0, inp1.size(0)):
			for j in range(0, inp1.size(2)):
				x = j
				y = 7
				z = 7
				inp2tmp_p.copy_(inp2)
				inp2tmp_m.copy_(inp2)
				inp2tmp_p[0,0,x,y,z] += self.eps
				inp2tmp_m[0,0,x,y,z] -= self.eps
				out1_p = self.vc(inp1, inp2tmp_p)
				out1_m = self.vc(inp1, inp2tmp_m)
				E1_p = torch.sum((out1_p-target)*(out1_p-target))
				E1_m = torch.sum((out1_m-target)*(out1_m-target))
								
				dE_dx = (E1_p.item() - E1_m.item())/(2.0*self.eps)
				# print(inp2.grad[0, 0, x, y, z].item(), dE_dx)
				self.assertLess(math.fabs(dE_dx - inp2.grad[0, 0, x, y, z].item()), math.fabs(dE_dx) * self.rtol + self.atol)
				# num_grads.append(dE_dx)
				# an_grads.append(inp2.grad[0,0,x,y,z].item())
		
		# import matplotlib.pylab as plt
		# fig = plt.figure()
		# plt.plot(num_grads, 'r.-', label = 'num grad')
		# plt.plot(an_grads,'bo', label = 'an grad')
		# plt.ylim(0, 100)
		# plt.legend()
		# plt.savefig('TestFig/test_backward_new.png')


if __name__=='__main__':
	unittest.main()