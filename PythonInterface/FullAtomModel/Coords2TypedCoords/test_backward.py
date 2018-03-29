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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Angles2Coords.Angles2Coords import Angles2Coords
from Coords2TypedCoords import Coords2TypedCoords

def test_gradient():
	sequence = ['GGMLGWAHFGY']
	x0 = Variable(torch.DoubleTensor(1,7,len(sequence[0])), requires_grad=True)
	x1 = Variable(torch.DoubleTensor(1,7,len(sequence[0])))
	x0.data[0,0,:] = -1.047
	x0.data[0,1,:] = -0.698
	x0.data[0,2:,:] = 110.4*np.pi/180.0
	
	a2c = Angles2Coords()
	c2tc = Coords2TypedCoords()
	
	
	y0, res, at = a2c(x0, sequence)
	coords, num_atoms_of_type, offsets = c2tc(y0, res, at, a2c.num_atoms)
	z0 = coords.sum()
		
	z0.backward()
	back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)
	
	for i in range(0,7):
		grads = []
		for j in range(0,x0.size(2)):
			dx = 0.0001
			x1.data.copy_(x0.data)
			x1.data[0,i,j]+=dx
			y1, res, at = a2c(x1, sequence)
			coords, num_atoms_of_type, offsets = c2tc(y1, res, at, a2c.num_atoms)
			z1 = coords.sum()
			dy_dx = (z1.data[0]-z0.data[0])/(dx)
			grads.append(dy_dx)

		fig = plt.figure()
		plt.plot(grads, 'r.-', label = 'num grad')
		plt.plot(back_grad_x0[0,i,:].numpy(),'bo', label = 'an grad')
		plt.legend()
		plt.savefig('TestFig/test_backward_%d.png'%i)

if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient()