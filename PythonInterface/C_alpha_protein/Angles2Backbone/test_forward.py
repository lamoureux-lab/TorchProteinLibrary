import sys
import os
import torch
from torch.autograd import Variable
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim

from angles2backbone import Angles2Backbone
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from FullAtomModel import Angles2Coords

if __name__=='__main__':
	L=10
	x0 = Variable(torch.randn(1, 2, L).float().cuda())
	# x0.data.fill_(0.0)
	length = Variable(torch.IntTensor(1).cuda().fill_(L))
	a2c = Angles2Backbone()
	proteins = a2c(x0, length)
	print proteins.size()
	proteins = proteins.data.cpu().resize_(1,3*L,3).numpy()
	
	a2cfa = Angles2Coords()
	x0 = x0.cpu()
	sequence = ''
	for i in xrange(L):
		sequence += 'G'
	print sequence
	x1 = Variable(torch.zeros(1, 7, L).double())
	x1.data[:,0:2,:].copy_(x0.data)
	print x1
	proteins_fa, atom_names, res_names, num_atoms = a2cfa(x1,[sequence])
	print num_atoms, proteins_fa.size(), atom_names.size()
	proteins_fa = proteins_fa.data.cpu().resize_(1,num_atoms.data[0],3).numpy()

	sx, sy, sz = proteins[0,:,0], proteins[0,:,1], proteins[0,:,2]
	rx, ry, rz = proteins_fa[0,:,0], proteins_fa[0,:,1], proteins_fa[0,:,2]
	fig = plt.figure()
	plt.title("Full atom model forward test")
	ax = p3.Axes3D(fig)
	ax.plot(sx,sy,sz, '-', label = 'a2c')
	ax.plot(rx,ry,rz, '.', label = 'ref')
	ax.set_xlim(-10,10)
	ax.set_ylim(-10,10)
	ax.set_zlim(-10,10)
	ax.legend()
	plt.show()



