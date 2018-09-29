import sys
import os
import torch
from torch.autograd import Variable
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim

from Angles2Backbone import Angles2Backbone

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from FullAtomModel import Angles2Coords

if __name__=='__main__':
	L=700
	x0 = torch.randn(1, 2, L, dtype=torch.float, device='cuda')
	length = torch.zeros(1, dtype=torch.int, device='cuda').fill_(L)
	a2c = Angles2Backbone()
	proteins = a2c(x0, length)
	proteins = proteins.data.cpu().resize_(1,3*L,3).numpy()
	
	a2cfa = Angles2Coords()
	x0 = x0.cpu()
	sequence = ''
	for i in range(L):
		sequence += 'A'
	
	x1 = torch.zeros(1, 7, L, dtype=torch.double)
	x1.data[:,0:2,:].copy_(x0.data)
	
	proteins_fa, res_names, atom_names, num_atoms = a2cfa(x1,[sequence])
	proteins_fa = proteins_fa.data.cpu().resize_(1,num_atoms.data[0],3).numpy()
	
	error = []
	k=0
	for i in range(num_atoms.data[0]):
		# print atom_names.data[0,i,0], atom_names.data[0,i,1], atom_names.data[0,i,2]
		if atom_names.data[0,i,0] == 67 and  atom_names.data[0,i,1] == 0: #C
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			k+=1
		if atom_names.data[0,i,0] == 67 and  atom_names.data[0,i,1] == 65 and  atom_names.data[0,i,2] == 0: #CA
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			k+=1
		if atom_names.data[0,i,0] == 78 and  atom_names.data[0,i,1] == 0: #N
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			k+=1
	
	if not os.path.exists("TestFig"):
		os.mkdir("TestFig")

	fig = plt.figure()
	plt.plot(error, '-', label = 'error')
	plt.savefig('TestFig/forward_precision.png')

	min_xyz = np.min(proteins_fa[0,:,:])
	max_xyz = np.max(proteins_fa[0,:,:])
	sx, sy, sz = proteins[0,:,0], proteins[0,:,1], proteins[0,:,2]
	rx, ry, rz = proteins_fa[0,:,0], proteins_fa[0,:,1], proteins_fa[0,:,2]
	fig = plt.figure()
	plt.title("Full atom model forward test")
	ax = p3.Axes3D(fig)
	ax.plot(rx,ry,rz, '.', label = 'ref')
	ax.plot(sx,sy,sz, '-', label = 'a2c')
	ax.set_xlim(min_xyz,max_xyz)
	ax.set_ylim(min_xyz,max_xyz)
	ax.set_zlim(min_xyz,max_xyz)
	ax.legend()
	plt.savefig('TestFig/forward_precision_trace.png')



