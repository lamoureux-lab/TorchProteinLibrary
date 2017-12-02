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

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Angles2CoordsAB import Angles2CoordsAB as Angles2Coords
from angles2BMatrixAB import Angles2BMatrixAB

def test_forward():
	L=4
	angles = Variable(torch.randn(2, L).cuda(), requires_grad=True)
	grad_output = Variable(torch.ones(3*(L+1)).cuda(), requires_grad=True)
	length = Variable(torch.IntTensor(1).fill_(L))
	
	a2c = Angles2Coords(L)

	coords = a2c(angles, length)
	coords.backward(grad_output)
	back_grad_coords = torch.FloatTensor(angles.grad.size()).copy_(angles.grad.data)
	
	back_grad_coords = coords.grad_fn.dr_dangle.resize_(2, L, L+1, 3)

	a2b = Angles2BMatrixAB(L)
	B = a2b.forward(angles, coords, length)
	B = B.data.resize_(2, L, L+1, 3)
	print B[0,:,:,:]
	print back_grad_coords[0,:,:,:]

	mat = np.zeros( (L, L) )
	for i in range(0,L):
		for j in range(0,L):
			for m in range(0, L+1):
				mat[i,j] += torch.sum(torch.mul(back_grad_coords[0,i,m,:], B[0,j,m,:]))
	
	print mat
	

	print back_grad_coords.sum()

def visualize_coordinates(coordinates, angles_length):
	import matplotlib.pylab as plt
	import mpl_toolkits.mplot3d.axes3d as p3
	import seaborn as sea
	L = angles_length+1
	coordinates = coordinates.data.cpu().resize_( coordinates.size(0), 3)
	rx = coordinates[:L,0].numpy()
	ry = coordinates[:L,1].numpy()
	rz = coordinates[:L,2].numpy()
	
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.plot(rx,ry,rz, '-', color='black', label = 'structure')
	
	ax.legend()
	plt.show()

def visualize_coordinates_it(ax, coordinates, angles_length):
	
	L = angles_length+1
	coordinates = coordinates.data.cpu().resize_( coordinates.size(0), 3)
	rx = coordinates[:L,0].numpy()
	ry = coordinates[:L,1].numpy()
	rz = coordinates[:L,2].numpy()
	
	
	ax.plot(rx,ry,rz, '-', color='black', label = 'structure')
	
	
def test_forward_iterative():
	L=4
	angles = Variable(torch.randn(2, L).cuda(), requires_grad=False)
	length = Variable(torch.IntTensor(1).fill_(L))
	dcoords = Variable(torch.zeros(L+1,3).cuda(), requires_grad=False)
	dangles = Variable(torch.zeros(2,L).cuda(), requires_grad=False)
	
	dcoords[4,0]=0.1
	dcoords[4,1]=0.1
	dcoords[4,2]=0.1

	a2c = Angles2Coords(L)
	a2b = Angles2BMatrixAB(L)
	
	fig = plt.figure()
	ax = p3.Axes3D(fig)

	coords = a2c(angles, length)
	visualize_coordinates_it(ax, coords, L)
	for i in range(0,10):
		coords = a2c(angles, length)
		B = a2b(angles, coords, length)
		B = B.resize(2, L, L+1, 3)
		
		for i in range(0,L):
			angles[0,i] = angles[0,i] + torch.sum(torch.mul(B[0,i,:,:], dcoords))
			angles[1,i] = angles[1,i] + torch.sum(torch.mul(B[1,i,:,:], dcoords))

		if i%100==0:
			visualize_coordinates_it(ax, coords, L)

	visualize_coordinates_it(ax, coords, L)

	ax.legend()
	plt.show()
		

if __name__=='__main__':
	test_forward_iterative()
	# test_forward()
	