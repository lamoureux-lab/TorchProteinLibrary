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

from TorchProteinLibrary.ReducedModel import Angles2Backbone as Angles2Coords

def test_gradient():
	L=10
	x0 = torch.zeros(1, 2, L, dtype=torch.float, device='cuda').normal_().requires_grad_()
	x1 = torch.zeros(1, 2, L, dtype=torch.float, device='cuda').normal_()
	length = torch.zeros(1, dtype=torch.int, device='cuda').fill_(L)
	
	model = Angles2Coords()
		
	basis_x0 = model(x0, length)
	err_x0 = basis_x0.sum()
	err_x0.backward()
	back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.float, device='cpu').copy_(x0.grad.data)
		
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
	plt.plot(grads[0],'--ro', label = 'num phi')
	plt.plot(grads[1],'-r', label = 'num psi')
	plt.plot(back_grad_x0[0,0,:].numpy(),'-b', label = 'an phi')
	plt.plot(back_grad_x0[0,1,:].numpy(),'--bo', label = 'an psi')
	
	plt.legend()
	plt.savefig('TestFig/angles2backbone_gradients.png')

def test_gradient_batch():
	L=10
	batch_size = 2
	x0 = torch.zeros(batch_size, 2, L, dtype=torch.float, device='cuda').normal_().requires_grad_()
	x1 = torch.zeros(batch_size, 2, L, dtype=torch.float, device='cuda').normal_()
	length = torch.zeros(batch_size, dtype=torch.int, device='cuda').fill_(L)
		
	model = Angles2Coords()

	basis_x0 = model(x0, length)
	err_x0 = basis_x0.sum()
	err_x0.backward()
	back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.float, device='cpu').copy_(x0.grad.data)
	
	for b in range(0,batch_size):
		grads = [[],[]]
		for a in range(0,2):
			for i in range(0,L):
				dx = 0.001
				x1.data.copy_(x0.data)
				x1.data[b,a,i]+=dx
				basis_x1 = model(x1, length)
				err_x1 = basis_x1.sum()
				derr_dangles = (err_x1-err_x0)/(dx)
				grads[a].append(derr_dangles.data[0])
		
		fig = plt.figure()
		plt.plot(grads[0],'-r', label = 'num alpha')
		plt.plot(grads[1],'--ro', label = 'num beta')
		plt.plot(back_grad_x0[b,0,:].numpy(),'--bo', label = 'an alpha')
		plt.plot(back_grad_x0[b,1,:].numpy(),'-b', label = 'an beta')
		plt.legend()
		plt.savefig('TestFig/angles2backbone_gradients_batch%d.png'%b)
		plt.clf()


if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient()
	test_gradient_batch()