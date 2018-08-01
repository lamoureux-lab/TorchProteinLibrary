import sys
import os
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

from VolumeConvolution import VolumeConvolution

def fill_delta(r0, volume):
	# volume.data.fill_(-0.1)
	volume.data[0,0,r0[0],r0[1],r0[2]] = 1

def test_gradient():
	
	vc = VolumeConvolution(num_features=1)
	vc.W.data.fill_(1.0)
	v_size = 30
	inp1 = Variable(torch.FloatTensor(1, 1, v_size, v_size, v_size).fill_(0.0).cuda(), requires_grad=True)
	inp1tmp = Variable(torch.FloatTensor(1, 1, v_size, v_size, v_size).fill_(0.0).cuda())
	inp2 = Variable(torch.FloatTensor(1, 1, v_size, v_size, v_size).fill_(0.0).cuda(), requires_grad=True)
	fill_delta(np.array([1,6,5]), inp1)
	fill_delta(np.array([5,5,5]), inp2)
	
	out = vc(inp1, inp2)
	E = out.sum()
	E.backward()
	
	back_grad_x0 = torch.FloatTensor(inp1.grad.size()).copy_(inp1.grad.data)
	inp1tmp.data.copy_(inp1.data)
	grads = []
	real_grads = []
	Np=10
	
	for j in range(0,Np):
		x = random.randint(0,v_size-1)
		y = random.randint(0,v_size-1)
		z = random.randint(0,v_size-1)
		dx = 0.0001
		inp1tmp.data.copy_(inp1.data)
		inp1tmp.data[0,0,x,y,z]+=dx
		E1 = vc(inp1tmp, inp2).sum()
		
		dy_dx = (E1.data[0]-E.data[0])/(dx)
		grads.append(dy_dx)
		real_grads.append(back_grad_x0[0,0,x,y,z])
		fig = plt.figure()
		plt.plot(grads, 'r.-', label = 'num grad')
		plt.plot(real_grads,'bo', label = 'an grad')
		plt.legend()
		plt.savefig('TestFig/test_backward.png')




if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient()

	# print out


