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

from angles2basisDihedral import Angles2BasisDihedral as Angles2Basis

	

def test_gradient():
	max_angles = 10
	max_atoms = 3*(max_angles+1)
	num_angles = 5
	num_atoms = 3*(num_angles+1)
	x0 = Variable(torch.randn(2, max_angles).cuda(), requires_grad=True)
	x1 = Variable(torch.randn(2, max_angles).cuda())
	length = Variable(torch.IntTensor(1).fill_(num_angles))
	
	target = Variable(torch.zeros(3, max_atoms).cuda())
	target.data.fill_(1.0)
	
	model = Angles2Basis(angles_max_length=max_angles).cuda()
	loss_fn = nn.MSELoss(False).cuda()

	basis_x0 = model(x0, length)
	err_x0 = loss_fn(basis_x0, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)
		
	grads = [[],[]]
	for a in range(0,2):
		for i in range(0,max_angles):
			dx = 0.001
			x1.data.copy_(x0.data)
			x1.data[a,i]+=dx
			basis_x1 = model(x1, length)
			err_x1 = loss_fn(basis_x1, target)
			derr_dangles = (err_x1-err_x0)/(dx)
			grads[a].append(derr_dangles.data[0])
	
	fig = plt.figure()
	plt.plot(grads[0],'--ro', label = 'num alpha')
	plt.plot(grads[1],'-r', label = 'num beta')
	plt.plot(back_grad_x0[0,:].numpy(),'-b', label = 'an alpha')
	plt.plot(back_grad_x0[1,:].numpy(),'--bo', label = 'an beta')
	plt.legend()
	plt.savefig('TestFig/angles2basisDihedral_gradients.png')

def test_gradient_batch():
	max_angles = 10
	max_atoms = 3*(max_angles+1)
	num_angles = 5
	num_atoms = 3*(num_angles+1)
	x0 = Variable(torch.randn(2, 2, max_angles).cuda(), requires_grad=True)
	x1 = Variable(torch.randn(2, 2, max_angles).cuda())
	length = Variable(torch.IntTensor(2).fill_(num_angles))
	
	target = Variable(torch.zeros(2, 3, max_atoms).cuda())
	target.data.fill_(1.0)
	
	model = Angles2Basis(angles_max_length=max_angles).cuda()

	loss_fn = nn.MSELoss(False).cuda()

	basis_x0 = model(x0, length)
	err_x0 = loss_fn(basis_x0, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)
		
	
	for b in range(0,2):
		grads = [[],[]]
		for a in range(0,2):
			for i in range(0,max_angles):
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
		plt.plot(back_grad_x0[b,1,:].numpy(),'-b', label = 'an beta')
		plt.legend()
		plt.savefig('TestFig/angles2basisDihedral_gradients_batch%d.png'%b)
		plt.clf()


if __name__=='__main__':
	test_gradient()
	test_gradient_batch()