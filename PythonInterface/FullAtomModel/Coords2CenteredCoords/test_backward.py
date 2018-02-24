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
from Coords2TypedCoords.Coords2TypedCoords import Coords2TypedCoords
from Coords2CenteredCoords import Coords2CenteredCoords

def test_gradient():
	sequence = 'GGMLGWAHFGY'
	x0 = Variable(torch.DoubleTensor(7,len(sequence)), requires_grad=True)
	x1 = Variable(torch.DoubleTensor(7,len(sequence)))
	x0.data[0,:] = -1.047
	x0.data[1,:] = -0.698
	x0.data[2:,:] = 110.4*np.pi/180.0
	
	a2c = Angles2Coords(sequence)
	c2tc = Coords2TypedCoords(num_atoms=a2c.num_atoms)
	c2cc = Coords2CenteredCoords(120)
	
	
	y0, res, at = a2c(x0)
	y0 = c2cc(y0)
	coords, num_atoms_of_type, offsets = c2tc(y0, res, at)
	z0 = coords.sum()
		
	z0.backward()
	back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)

	x1.data.copy_(x0.data)
	y1, res, at = a2c(x1)
	coords, num_atoms_of_type, offsets = c2tc(y1, res, at)
	z0 = coords.sum()

	for i in range(0,7):
		grads = []
		for j in range(0,x0.size(1)):
			dx = 0.0001
			x1.data.copy_(x0.data)
			x1.data[i,j]+=dx
			y1, res, at = a2c(x1)
			coords, num_atoms_of_type, offsets = c2tc(y1, res, at)
			z1 = coords.sum()
			dy_dx = (z1.data[0]-z0.data[0])/(dx)
			grads.append(dy_dx)

		fig = plt.figure()
		plt.plot(grads, 'r.-', label = 'num grad')
		plt.plot(back_grad_x0[i,:].numpy(),'bo', label = 'an grad')
		plt.legend()
		plt.savefig('TestFig/test_backward_%d.png'%i)

if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient()