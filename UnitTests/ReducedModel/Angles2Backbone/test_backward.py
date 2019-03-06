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
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim

from TorchProteinLibrary.ReducedModel import Angles2Backbone as Angles2Coords

def test_gradient(device = 'cuda'):
	L=65
	x0 = torch.zeros(1, 3, L, dtype=torch.float, device=device).normal_().requires_grad_()
	x1 = torch.zeros(1, 3, L, dtype=torch.float, device=device).normal_()
	length = torch.zeros(1, dtype=torch.int, device=device).fill_(L)
	
	model = Angles2Coords()
		
	basis_x0 = model(x0, length)
	err_x0 = basis_x0.sum()
	err_x0.backward()
	
	
	with torch.no_grad():
		back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.float, device='cpu').copy_(x0.grad)
		grads = [[],[],[]]
		for a in range(0,3):
			for i in range(0,L):
				dx = 0.0001
				x1.copy_(x0.data)
				x1[0,a,i]+=dx
				basis_x1 = model(x1, length)
				err_x1 = basis_x1.sum()
				derr_dangles = (err_x1-err_x0)/(dx)
				grads[a].append(derr_dangles.item())
	
	fig = plt.figure()
	plt.plot(grads[0],'-r', label = 'num phi')
	plt.plot(grads[1],'-g', label = 'num psi')
	plt.plot(grads[2],'-b', label = 'num omega')
	plt.plot(back_grad_x0[0,0,:].numpy(),'--ro', label = 'an phi')
	plt.plot(back_grad_x0[0,1,:].numpy(),'--go', label = 'an psi')
	plt.plot(back_grad_x0[0,2,:].numpy(),'--bo', label = 'an omega')
	
	plt.legend()
	plt.savefig('TestFig/angles2backbone_gradients.png')

def test_gradient_batch(device='cuda'):
	L=65
	batch_size = 10
	x0 = torch.zeros(batch_size, 3, L, dtype=torch.float, device=device).normal_().requires_grad_()
	x1 = torch.zeros(batch_size, 3, L, dtype=torch.float, device=device).normal_()
	length = torch.zeros(batch_size, dtype=torch.int, device=device).fill_(L)
		
	model = Angles2Coords()

	basis_x0 = model(x0, length)
	err_x0 = basis_x0.sum()
	err_x0.backward()
		
	with torch.no_grad():
		back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.float, device='cpu').copy_(x0.grad)
		for b in range(0,batch_size):
			grads = [[],[],[]]
			for a in range(0,3):
				for i in range(0,L):
					dx = 0.0001
					x1.copy_(x0)
					x1[b,a,i]+=dx
					basis_x1 = model(x1, length)
					err_x1 = basis_x1.sum()
					derr_dangles = (err_x1-err_x0)/(dx)
					grads[a].append(derr_dangles.item())

			fig = plt.figure()
			plt.plot(grads[0],'-r', label = 'num alpha')
			plt.plot(grads[1],'-g', label = 'num beta')
			plt.plot(grads[2],'-b', label = 'num omega')
			plt.plot(back_grad_x0[b,0,:].numpy(),'--ro', label = 'an alpha')
			plt.plot(back_grad_x0[b,1,:].numpy(),'--go', label = 'an beta')
			plt.plot(back_grad_x0[b,2,:].numpy(),'--bo', label = 'an omega')
			plt.legend()
			plt.savefig('TestFig/angles2backbone_gradients_batch%d.png'%b)
			plt.clf()


if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient('cpu')
	test_gradient_batch('cpu')