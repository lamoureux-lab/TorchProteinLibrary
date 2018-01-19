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

if __name__=='__main__':

	sequence = 'GGGGGG'
	angles = Variable(torch.DoubleTensor(len(sequence),7).zero_())
	angles[:,0] = -1.047
	angles[:,1] = -0.698
	angles[:,2:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	protein = a2c(sequence, angles)
	protein = protein.data.resize_(protein.size(0)/3, 3).numpy()
	
	# print protein
	sx, sy, sz = protein[:,0], protein[:,1], protein[:,2]
	fig = plt.figure()
	plt.title("Full atom model forward test")
	ax = p3.Axes3D(fig)
	ax.plot(sx,sy,sz, 'r.', label = 'atoms')
	ax.legend()
	plt.show()