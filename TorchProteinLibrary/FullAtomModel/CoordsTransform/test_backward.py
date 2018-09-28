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
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Angles2Coords.Angles2Coords import Angles2Coords
from Coords2TypedCoords.Coords2TypedCoords import Coords2TypedCoords
from CoordsTransform import Coords2CenteredCoords

def test_gradient():
	sequence = ['GGMLGWAHFGY']
	x0 = Variable(torch.zeros(1, 7, len(sequence[0]), dtype=torch.double), requires_grad=True)
	x1 = Variable(torch.zeros(1, 7, len(sequence[0]), dtype=torch.double))
	x0.data[0,0,:] = -1.047
	x0.data[0,1,:] = -0.698
	x0.data[0,2:,:] = 110.4*np.pi/180.0
	
	a2c = Angles2Coords()
	c2tc = Coords2TypedCoords()
	c2cc = Coords2CenteredCoords(rotate=True, translate=True)
	
	
	y0, res, at, num_atoms = a2c(x0, sequence)
	y0 = c2cc(y0, num_atoms)
	coords, num_atoms_of_type, offsets = c2tc(y0, res, at, num_atoms)
	coords = coords.resize(1, int(coords.size(1)/3), 3)
	center_mass = coords.mean(dim=1).unsqueeze(dim=1)
	coords = coords - center_mass
	Rg = torch.mean(torch.sqrt((coords*coords).sum(dim=2)))	
	Rg.backward()
	back_grad_x0 = torch.zeros(x0.grad.size(), dtype=torch.double).copy_(x0.grad.data)

	x1.data.copy_(x0.data)
	
	for i in range(0,7):
		grads = []
		for j in range(0,x0.size(2)):
			dx = 0.0001
			x1.data.copy_(x0.data)
			x1.data[0,i,j]+=dx
			y1, res, at, num_atoms = a2c(x1, sequence)
			coords, num_atoms_of_type, offsets = c2tc(y1, res, at, num_atoms)
			coords = coords.resize(1, int(coords.size(1)/3), 3)
			center_mass = coords.mean(dim=1).unsqueeze(dim=1)
			coords = coords - center_mass
			Rg_1 = torch.mean(torch.sqrt((coords*coords).sum(dim=2)))	
			
			dy_dx = (Rg_1.data-Rg.data)/(dx)
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