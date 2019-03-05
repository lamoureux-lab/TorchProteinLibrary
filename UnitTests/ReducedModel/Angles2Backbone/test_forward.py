import sys
import os
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sea

from matplotlib import pylab as plt
import pandas as pd
from tqdm import tqdm
import pickle as pkl

from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.ReducedModel import Angles2Backbone

def measure_trace(length=700, device='cuda'):
	x0 = torch.randn(1, 3, length, dtype=torch.float, device=device)
	x0.data[:,2,:].fill_(-3.1318)
	length = torch.zeros(1, dtype=torch.int, device=device).fill_(length)
	a2c = Angles2Backbone()
	proteins = a2c(x0, length)
	proteins = proteins.data.cpu().resize_(1,3*length,3).numpy()
	
	a2cfa = Angles2Coords()
	x0 = x0.cpu()
	sequence = ''.join(['A' for i in range(length)])
	
	x1 = torch.zeros(1, 7, length, dtype=torch.double)
	x1.data[:,0:2,:].copy_(x0.data[:,0:2,:])
	
	proteins_fa, res_names, atom_names, num_atoms = a2cfa(x1,[sequence])
	proteins_fa = proteins_fa.data.cpu().resize_(1,num_atoms.data[0],3).numpy()
	
	error = []
	index = []
	k=0
	for i in range(num_atoms.data[0]):
		if atom_names.data[0,i,0] == 67 and  atom_names.data[0,i,1] == 0: #C
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			index.append(k)
			k+=1
		if atom_names.data[0,i,0] == 67 and  atom_names.data[0,i,1] == 65 and  atom_names.data[0,i,2] == 0: #CA
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			index.append(k)
			k+=1
		if atom_names.data[0,i,0] == 78 and  atom_names.data[0,i,1] == 0: #N
			error.append( np.linalg.norm(proteins[0,k,:] - proteins_fa[0,i,:]))
			index.append(k)
			k+=1

	return error, index

def measure_statistics(length=700, num_measurements=10, output_filename='ErrorStat.pkl', device='cuda'):
	errors = []
	indexes = []
	for i in range(num_measurements):
		error, index = measure_trace(length, device=device)
		errors += error
		indexes += index
	
	data = pd.DataFrame({  	'Index': indexes, 
							'Error': errors
						})
	data.to_pickle(output_filename)

if __name__=='__main__':
		
	if not os.path.exists("TestFig"):
		os.mkdir("TestFig")
	
	measure_statistics(length=700, num_measurements=3, output_filename='ErrorStatOmega.pkl', device='cpu')
	
	data = pd.read_pickle('ErrorStatOmega.pkl')
	sea.set_style("whitegrid")
	sea.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
	g1 = sea.relplot(x="Index", y="Error", kind="line", height=6, aspect=1.3, markers=True, data=data)
	plt.ylabel("Error, Angstroms")
	plt.xlabel("Atomic index")
	plt.xlim(0,2100)
	sea.despine()
	# plt.show()
	plt.savefig("TestFig/ErrorStat.png")


	# fig = plt.figure()
	# plt.plot(error, '-', label = 'error')
	# plt.savefig('TestFig/forward_precision.png')

	# min_xyz = np.min(proteins_fa[0,:,:])
	# max_xyz = np.max(proteins_fa[0,:,:])
	# sx, sy, sz = proteins[0,:,0], proteins[0,:,1], proteins[0,:,2]
	# rx, ry, rz = proteins_fa[0,:,0], proteins_fa[0,:,1], proteins_fa[0,:,2]
	# fig = plt.figure()
	# plt.title("Full atom model forward test")
	# ax = p3.Axes3D(fig)
	# ax.plot(rx,ry,rz, '.', label = 'ref')
	# ax.plot(sx,sy,sz, '-', label = 'a2c')
	# ax.set_xlim(min_xyz,max_xyz)
	# ax.set_ylim(min_xyz,max_xyz)
	# ax.set_zlim(min_xyz,max_xyz)
	# ax.legend()
	# plt.savefig('TestFig/forward_precision_trace.png')



