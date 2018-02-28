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

def batch_test():
	# sequences = ['GGGGGG', 'GGMLGWAHFGY']
	sequences = ['GGMLGWAHFGY', 'GGGGGG']
	# sequences = ['A']
	
	angles = Variable(torch.DoubleTensor(len(sequences), 7, len(sequences[0])).zero_())
	angles[:, 0, :] = -1.047
	angles[:, 1, :] = -0.698
	angles[:, 2:,:] = 110.4*np.pi/180.0
	
	a2c = Angles2Coords()
	protein, res_names, atom_names = a2c(angles, sequences)
	proteins = protein.data.resize_(len(sequences), protein.size(1)/3, 3).numpy()
	
	for i in range(0, len(sequences)):
		for j in range(0, a2c.num_atoms[i]):
			print j, res_names.data[i,j,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0], atom_names.data[i,j,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0]

	# print protein
	for i in range(0, len(sequences)):
		sx, sy, sz = proteins[i,:,0], proteins[i,:,1], proteins[i,:,2]
		fig = plt.figure()
		plt.title("Full atom model forward test")
		ax = p3.Axes3D(fig)
		ax.plot(sx,sy,sz, 'r.', label = 'atoms')
		ax.legend()
		plt.show()

if __name__=='__main__':
	batch_test()
	