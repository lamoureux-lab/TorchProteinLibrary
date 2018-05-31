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

from angles2coordsDihedral import Angles2CoordsDihedral as Angles2Coords

def test_gradient():
	L=10
	x0 = Variable(torch.DoubleTensor(1, 2, L).normal_().cuda(), requires_grad=True)
	x1 = Variable(torch.DoubleTensor(1, 2, L).normal_().cuda())
	length = Variable(torch.IntTensor(1).fill_(L).cuda())
	
	target = Variable(torch.DoubleTensor(1, 3*L).normal_().cuda())
	# target.data.fill_(1.0)
	
	model = Angles2Coords()
	# loss_fn = nn.MSELoss(False).cuda()
	
	basis_x0 = model(x0, length)
	err_x0 = basis_x0.sum()
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)
	# sys.exit()
		
	grads = [[],[]]
	for a in range(0,2):
		for i in range(0,L):
			dx = 0.001
			x1.data.copy_(x0.data)
			x1.data[0,a,i]+=dx
			basis_x1 = model(x1, length)
			err_x1 = basis_x1.sum()
			derr_dangles = (err_x1-err_x0)/(dx)
			grads[a].append(derr_dangles.data[0])
	
	fig = plt.figure()
	plt.plot(grads[0],'--ro', label = 'num alpha')
	plt.plot(grads[1],'-r', label = 'num beta')
	plt.plot(back_grad_x0[0,0,:].numpy(),'-b', label = 'an alpha')
	plt.plot(back_grad_x0[0,1,:].numpy(),'--bo', label = 'an beta')
	plt.legend()
	plt.savefig('TestFig/angles2coordsDihedral_gradients.png')

def test_gradient_batch():
	L=10
	x0 = Variable(torch.randn(2, 2, L).cuda(), requires_grad=True)
	x1 = Variable(torch.randn(2, 2, L).cuda())
	length = Variable(torch.IntTensor(2).fill_(L))
	
	target = Variable(torch.zeros(2, 3*(L+1)).cuda())
	target.data.fill_(1.0)
	
	model = TestNet(maxlen=L).cuda()

	loss_fn = nn.MSELoss(False).cuda()

	basis_x0 = model(x0, length)
	err_x0 = loss_fn(basis_x0, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)
		
	
	for b in range(0,2):
		grads = [[],[]]
		for a in range(0,2):
			for i in range(0,L-1):
				dx = 0.001
				x1.data.copy_(x0.data)
				x1.data[b,a,i]+=dx
				basis_x1 = model(x1, length)
				err_x1 = loss_fn(basis_x1, target)
				derr_dangles = (err_x1-err_x0)/(dx)
				grads[a].append(derr_dangles.data[0])
		
		fig = plt.figure()
		plt.plot(grads[0],'-r', label = 'num alpha')
		plt.plot(grads[1],'--ro', label = 'num beta')
		plt.plot(back_grad_x0[b,0,:].numpy(),'--bo', label = 'an alpha')
		plt.plot(back_grad_x0[b,1,:].numpy(),'-b', label = 'num beta')
		plt.legend()
		plt.savefig('TestFig/angles2coordsDihedral_gradients_batch%d.png'%b)
		plt.clf()


if __name__=='__main__':
	test_gradient()
	# test_gradient_batch()