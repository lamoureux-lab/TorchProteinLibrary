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

from coords2pairs import Coords2Pairs

def test_forward():
	max_coords = 36
	max_atoms = max_coords/3
	num_coords = 18
	num_atoms = num_coords/3
	
	x0 = Variable(torch.randn(max_coords).cuda(), requires_grad=True)
	length = Variable(torch.IntTensor(1).fill_(num_coords/3 - 1))
	
	target = Variable(torch.zeros(3*max_atoms*max_atoms).cuda())
	target.data.fill_(1.0)
	
	model = Coords2Pairs(angles_max_length=max_coords/3 - 1).cuda()
	pairs = model(x0, length)

	
	fig = plt.figure()
	plt.title("Pairs")
	pairs = torch.FloatTensor(pairs.size()).copy_(pairs.data)
	pairs.resize_(3, max_atoms, max_atoms)
	pp = torch.sqrt(pairs[0,:,:]*pairs[0,:,:] + pairs[1,:,:]*pairs[1,:,:] + pairs[2,:,:]*pairs[2,:,:])
	pp = pp.numpy()
	plt.imshow(pp)
	plt.colorbar()
	plt.show()

	mat = np.zeros((max_atoms, max_atoms))
	for i in range(0,max_atoms):
		for j in range(0,max_atoms):
			xi = x0[i*3].data[0]
			yi = x0[i*3+1].data[0]
			zi = x0[i*3+2].data[0]
			xj = x0[j*3].data[0]
			yj = x0[j*3+1].data[0]
			zj = x0[j*3+2].data[0]
			d = np.sqrt( (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj) )
			if i<num_atoms and j<num_atoms:
				mat[i,j]=d
			# print d, pp[j,i]

	print "Error:", np.sum(np.abs(pp-mat))


def test_gradient():
	
	max_coords = 36
	max_atoms = max_coords/3
	num_coords = 18
	
	x0 = Variable(torch.randn(max_coords).cuda(), requires_grad=True)
	x1 = Variable(torch.randn(max_coords).cuda())
	length = Variable(torch.IntTensor(1).fill_(num_coords/3 - 1))
	
	target = Variable(torch.zeros(3*max_atoms*max_atoms).cuda())
	target.data.fill_(1.0)
	
	model = Coords2Pairs(angles_max_length=max_coords/3 - 1).cuda()
	loss_fn = nn.MSELoss(False).cuda()

	basis_x0 = model(x0, length)
	err_x0 = loss_fn(basis_x0, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)

	grads = []
	for i in range(0,max_coords):
		dx = 0.001
		x1.data.copy_(x0.data)
		x1.data[i]+=dx
		basis_x1 = model(x1, length)
		err_x1 = loss_fn(basis_x1, target)
		derr_dangles = (err_x1-err_x0)/(dx)
		grads.append(derr_dangles.data[0])
	
	fig = plt.figure()
	plt.plot(grads,'--ro', label = 'num')
	plt.plot(back_grad_x0[:].numpy(),'-b', label = 'an')
	plt.legend()
	plt.savefig('TestFig/coords2pairs_gradients.png')

def test_gradient_batch():
	max_coords = 36
	max_atoms = max_coords/3
	num_coords = 18
	
	x0 = Variable(torch.randn(2, max_coords).cuda(), requires_grad=True)
	x1 = Variable(torch.randn(2, max_coords).cuda())
	length = Variable(torch.IntTensor(2).fill_(num_coords/3 - 1))
	
	target = Variable(torch.zeros(2, 3*max_atoms*max_atoms).cuda())
	target.data.fill_(1.0)
	
	model = Coords2Pairs(angles_max_length=max_coords/3 - 1).cuda()
	loss_fn = nn.MSELoss(False).cuda()

	basis_x0 = model(x0, length)
	err_x0 = loss_fn(basis_x0, target)
	err_x0.backward()
	back_grad_x0 = torch.FloatTensor(x0.grad.size()).copy_(x0.grad.data)
		
	
	for b in range(0,2):
		grads = []
		for i in range(0,max_coords):
			dx = 0.001
			x1.data.copy_(x0.data)
			x1.data[b,i]+=dx
			basis_x1 = model(x1, length)
			err_x1 = loss_fn(basis_x1, target)
			derr_dangles = (err_x1-err_x0)/(dx)
			grads.append(derr_dangles.data[0])

		fig = plt.figure()
		plt.plot(grads,'--ro', label = 'num')
		plt.plot(back_grad_x0[b,:].numpy(),'-b', label = 'an')
		plt.legend()
		plt.savefig('TestFig/coords2pairs_gradients_batch%d.png'%b)
		plt.clf()


if __name__=='__main__':
	test_forward()
	# test_gradient()
	# test_gradient_batch()