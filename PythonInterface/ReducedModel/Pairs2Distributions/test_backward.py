import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pylab as plt
plt.ioff()
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim
from random import randint
from pairs2distributions import Pairs2Distributions

def test_backward_nonvectorized():
	
	max_atoms = 30
	max_angles = max_atoms-1
	num_atoms = 30
	num_angles = num_atoms-1
	
	num_types = 3
	num_bins = 10
	resolution = 1.0
	vectors = np.random.rand(num_atoms,3)*10
	pdist = torch.zeros(3*max_atoms*max_atoms)
	
	plane_stride = max_atoms*max_atoms
	for i in range(0,num_atoms):
		for j in range(0,num_atoms):
			pdist[i*max_atoms + j] = (vectors[i,0] - vectors[j,0])
			pdist[plane_stride + i*max_atoms + j] = (vectors[i,1] - vectors[j,1])
			pdist[2*plane_stride + i*max_atoms + j] = (vectors[i,2] - vectors[j,2])
			
	
	x0 = Variable(pdist.cuda(), requires_grad=True)
	x1 = Variable(torch.randn(3*max_atoms*max_atoms).cuda())
	length = Variable(torch.IntTensor(1).fill_(num_angles))
	types = Variable(torch.IntTensor(max_atoms).fill_(0).cuda())
	for i in range(0,max_atoms):
		types[i] = randint(0, num_types-1)

	model = Pairs2Distributions(angles_max_length=max_angles, num_types=num_types, num_bins=num_bins, resolution=resolution).cuda()
	
	distr = model(x0, types, length)
	target = Variable(torch.zeros(distr.size()).cuda())
	loss_fn = nn.MSELoss(False).cuda()
	err_x0 = loss_fn(distr, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)

	grads = []
	for i in range(0,3*max_atoms*max_atoms):
		dx = 0.01
		x1.data.copy_(x0.data)
		x1.data[i]+=dx
		basis_x1 = model(x1, types, length)
		err_x1 = loss_fn(basis_x1, target)
		derr_dangles = (err_x1-err_x0)/(dx)
		grads.append(derr_dangles.data[0])
	
	fig = plt.figure()
	plt.plot(grads[0:100],'--ro', label = 'num')
	plt.plot(back_grad_x0.numpy()[0:100],'-b', label = 'an')
	plt.legend()
	plt.savefig('TestFig/pairs2dist_input_gradients.png')

	

if __name__=='__main__':
	test_backward_nonvectorized()