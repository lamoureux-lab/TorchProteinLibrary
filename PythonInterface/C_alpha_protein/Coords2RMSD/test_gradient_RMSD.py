import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim
from coords2RMSD import Coords2RMSD
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Angles2CoordsDihedral import Angles2CoordsDihedral

def test_gradient():
	maxL=2
	L=3
	x0 = Variable(torch.DoubleTensor([[2,0,0, 3,0.1,1, 3,0.1,0 ]]).cuda(), requires_grad=True)
	x1 = Variable(torch.DoubleTensor(x0.size()).random_().cuda())
	length = Variable(torch.IntTensor(1).fill_(L).cuda())
	target = Variable(torch.DoubleTensor([[0,0,0, 1,1,0, 3,0,0]]).cuda())

	loss = Coords2RMSD()
	rmsd_x0 = loss(x0, target, length)
	print rmsd_x0
	rmsd_x0.backward()
	float_rmsd_x0 = np.sqrt(rmsd_x0.data[0])
	# float_rmsd_x0 = rmsd_x0.data[0]
	back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)
	
	aligned_x0 = torch.zeros(3*maxL)
	aligned_target = torch.zeros(3*maxL)
	
	grads = []
	for i in range(0,3*L):
		dx = 0.001
		x1.data.copy_(x0.data)
		x1.data[0,i]+=dx
		rmsd_x1 = loss(x1, target, length)
		float_rmsd_x1 = np.sqrt(rmsd_x1.data[0])
		# float_rmsd_x1 = rmsd_x1.data[0]
		drmsd_dx = (float_rmsd_x1-float_rmsd_x0)/(dx)
		grads.append(drmsd_dx)
	
	fig = plt.figure()
	plt.plot(grads,'-r', label = 'num grad')
	plt.plot(back_grad_x0[0,:3*L].numpy(),'bo', label = 'an grad')
	plt.legend()
	plt.savefig('TestFig/rmsd_gradients_CUDA.png')

def test_gradient_combined():
	L = 10
	batch_size = 5
	angles0 = Variable(torch.DoubleTensor(batch_size, 2, L).normal_().cuda(), requires_grad=True)
	angles1 = Variable(torch.DoubleTensor(batch_size, 2, L).normal_().cuda())
	length = Variable(torch.IntTensor(batch_size).fill_(L).cuda())
	target = Variable(torch.DoubleTensor(batch_size, L*3).normal_().cuda())
	
	model = Angles2CoordsDihedral()
	loss = Coords2RMSD()
	
	x0 = model(angles0, length)
	rmsd_x0 = loss(x0, target, length)

	rmsd_x0.sum().backward()
	
	float_rmsd_x0 = torch.sqrt(rmsd_x0).sum().data[0]
	back_grad_angles0 = torch.DoubleTensor(angles0.grad.size()).copy_(angles0.grad.data)
	
	print float_rmsd_x0

	
	for b in xrange(batch_size):
		grads = [[],[]]
		for i in range(0,L):
			for k in range(0,2):
				dx = 0.0001
				angles1.data.copy_(angles0.data)
				angles1.data[b,k,i]+=dx
				x1 = model(angles1, length)
				rmsd_x1 = loss(x1, target, length)
				float_rmsd_x1 = torch.sqrt(rmsd_x1).sum().data[0]
				
				drmsd_dx = (float_rmsd_x1-float_rmsd_x0)/(dx)
				grads[k].append(drmsd_dx)
		
		fig = plt.figure()
		plt.plot(grads[0],'-r', label = 'num grad')
		plt.plot(back_grad_angles0[b,0,:].numpy(),'-.b', label = 'an grad')
		plt.plot(grads[1],'-r', label = 'num grad')
		plt.plot(back_grad_angles0[b,1,:].numpy(),'-.b', label = 'an grad')
		plt.legend()
		plt.savefig('TestFig/rmsd_gradients_CUDA_batch%d.png'%b)

if __name__=='__main__':
	# test_gradient()
	test_gradient_combined()