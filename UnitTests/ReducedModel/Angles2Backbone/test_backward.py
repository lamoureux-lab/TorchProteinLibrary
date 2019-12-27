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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(3)
np.random.seed(3)

def test_gradient(device = 'cpu', dtype=torch.double):
	L=65
	angles = torch.zeros(1, 3, L, dtype=dtype, device=device).normal_()
	length = torch.zeros(1, dtype=torch.int, device=device).fill_(L)
	
	model = Angles2Coords()
	param_x0 = model.get_default_parameters().to(device=device, dtype=dtype).requires_grad_()
	param_x1 = model.get_default_parameters().to(device=device, dtype=dtype).requires_grad_()
		
	y0 = model(angles, param_x0, length)
	err_y0 = y0.sum()
	err_y0.backward()
		
	with torch.no_grad():
		back_grad_x0 = torch.zeros(param_x0.grad.size(), dtype=dtype, device='cpu').copy_(param_x0.grad)
		grads = []
		for i in range(0,6):
			dx = 0.0001
			param_x1.copy_(param_x0.data)
			param_x1[i]+=dx
			y1 = model(angles, param_x1, length)
			err_y1 = y1.sum()
			derr_dangles = (err_y1-err_y0)/(dx)
			grads.append(derr_dangles.item())

	fig = plt.figure()
	
	plt.plot(back_grad_x0.numpy(),'--bo', label = 'an grad')
	plt.plot(grads,'-r', label = 'num grad')
	
	plt.legend()
	plt.savefig('TestFig/angles2backbone_param_gradients.png')

if __name__=='__main__':
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	test_gradient('cpu')
	