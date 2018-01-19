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

from Angles2Coords import Angles2Coords

def test_gradient():
	sequence = 'GG'
	x0 = Variable(torch.DoubleTensor(len(sequence),7), requires_grad=True)
	x1 = Variable(torch.DoubleTensor(len(sequence),7))
	x0.data[:,0] = -1.047
	x0.data[:,1] = -0.698
	x0.data[:,2:] = 110.4*np.pi/180.0
	
	a2c = Angles2Coords(sequence)
	y0 = a2c(x0).sum()
	
	y0.backward()
	back_grad_x0 = torch.DoubleTensor(x0.grad.size()).copy_(x0.grad.data)
	
	
	for i in range(0,2):
		grads = []
		for j in range(0,x0.size(1)):
			dx = 0.001
			x1.data.copy_(x0.data)
			x1.data[i,j]+=dx
			y1 = a2c(x1).sum()
			dy_dx = (y1.data[0]-y0.data[0])/(dx)
			grads.append(dy_dx)

		fig = plt.figure()
		plt.plot(grads,'-r', label = 'num grad')
		plt.plot(back_grad_x0[i,:].numpy(),'bo', label = 'an grad')
		plt.legend()
		plt.savefig('TestFig/test_backward_%d.png'%i)

if __name__=='__main__':
	test_gradient()