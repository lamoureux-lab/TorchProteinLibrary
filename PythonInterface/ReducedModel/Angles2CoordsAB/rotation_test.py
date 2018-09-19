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

from angles2coordsAB import Angles2CoordsAB as Angles2Coords

def test_mover():
	L=10
	angles = Variable(torch.randn(2, L).cuda())
	angles_mover = Variable(torch.randn(2, L).cuda().copy_(angles.data))
	angles_length = Variable(torch.IntTensor(1).fill_(L))
		
	a2c = Angles2Coords(angles_max_length=L).cuda()

	
	coords = a2c(angles, angles_length)

	angles.data[1,5] += 1.0
	angles.data[1,6] -= 1.0
	coords_moved = a2c(angles, angles_length)
	
	coords = coords.data.resize_( L+1, 3).cpu()
	coords_moved = coords_moved.data.resize_( L+1, 3).cpu()
	rx = coords[:L+1,0].numpy()
	ry = coords[:L+1,1].numpy()
	rz = coords[:L+1,2].numpy()
	mx = coords_moved[:L+1,0].numpy()
	my = coords_moved[:L+1,1].numpy()
	mz = coords_moved[:L+1,2].numpy()
	
	fig = plt.figure()
	plt.title("Fitted C-alpha model and the protein C-alpha coordinates")
	ax = p3.Axes3D(fig)
	ax.plot(rx,ry,rz, '--', color='black', label = 'init')
	ax.plot(mx,my,mz, '-', color='red', label = 'moved')
	
	ax.legend()	
	plt.show()
	



if __name__=='__main__':
	test_mover()
	