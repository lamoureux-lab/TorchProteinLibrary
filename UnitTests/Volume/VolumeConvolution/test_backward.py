import sys
import os

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np
import seaborn as sea
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

from TorchProteinLibrary.Volume import VolumeConvolution
import _Volume

def fill_V1(r0, r1, R0, R1, volume):
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

def fill_V2(r0, R0, volume):
	volume_size = volume.size(2)
	for x in range(volume_size):
		for y in range(volume_size):
			for z in range(volume_size):
				r_0 = np.array([x,y,z]) - r0
				dist0 = np.linalg.norm(r_0)
				if dist0 < R0:
					volume.data[0,0,x,y,z] = 1.0

def get_boundary(volume_in, volume_out):
	volume_size = volume_in.size(2)
	for x in range(1,volume_size-1):
		for y in range(1,volume_size-1):
			for z in range(1,volume_size-1):
				cs = torch.sum(volume_in.data[0,0, x-1:x+2, y-1:y+2, z-1:z+2]).item()
				cc = volume_in.data[0,0,x,y,z].item()
				if (cc<0.01) and (cs>0.9):
					volume_out.data[0,0,x,y,z] = 1.0


def test_gradient():
	
	vc = VolumeConvolution()
	v_size = 30
	inp1 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda', requires_grad=True)
	inp1tmp_p = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	inp1tmp_m = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	inp2 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda', requires_grad=True)

	target = torch.zeros(1, 1, 2*v_size, 2*v_size, 2*v_size, dtype=torch.float, device='cuda')
	fill_V2(np.array([12, 30, 50]), 5, target)

	r0 = [20,15,15]
	r1 = [23,15,7]
	l0 = [7,24,7]
	fill_V1(np.array(r0), np.array(r1), 8, 5, inp1)
	fill_V2(np.array(l0), 5, inp2)

	_Volume.Volume2Xplor(inp1.squeeze().cpu(), "receptor.xplor", 1.0)
	_Volume.Volume2Xplor(inp2.squeeze().cpu(), "ligand.xplor", 1.0)

	out = vc(inp1, inp2)
	E = torch.sum((out-target)*(out-target))
	# E = torch.sum(torch.abs(out-target))
	# E = out[:,:,8,40,0] + out[:,:,10,30,25]
	E.backward()
	
	back_grad_x0 = torch.zeros(inp2.grad.size(), dtype=torch.float, device='cpu').copy_(inp2.grad)

	print("Grad max min", torch.max(back_grad_x0), torch.min(back_grad_x0))
	_Volume.Volume2Xplor(back_grad_x0[0,0,:,:,:], "grad_conv.xplor", 1.00)
		
	num_grads = []
	an_grads = []	
	for j in range(0,v_size):
		x = j
		y = 7
		z = 7
		dx = 0.1
		inp1tmp_p.copy_(inp2)
		inp1tmp_m.copy_(inp2)
		inp1tmp_p[0,0,x,y,z]+=dx
		inp1tmp_m[0,0,x,y,z]-=dx
		
		# out1_p = vc(inp1tmp_p, inp2)
		# out1_m = vc(inp1tmp_m, inp2)
		out1_p = vc(inp1, inp1tmp_p)
		out1_m = vc(inp1, inp1tmp_m)
		E1_p = torch.sum((out1_p-target)*(out1_p-target))
		E1_m = torch.sum((out1_m-target)*(out1_m-target))
		# E1_p = torch.sum(torch.abs(out1_p-target))
		# E1_m = torch.sum(torch.abs(out1_m-target))
		# E1 = (out1[:,:,8,40,0] + out1[:,:,10,30,25])
		dy_dx = (E1_p.item()-E1_m.item())/(2*dx)

		num_grads.append(dy_dx)
		an_grads.append(back_grad_x0[0,0,x,y,z].item())
	
	fig = plt.figure()
	plt.plot(num_grads, 'r.-', label = 'num grad')
	plt.plot(an_grads,'bo', label = 'an grad')
	plt.legend()
	plt.savefig('TestFig/test_backward.png')

if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient()



