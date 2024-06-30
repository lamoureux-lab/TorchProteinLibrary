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
# import seaborn as sea

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import Coords2Center, CoordsTranslate
from TorchProteinLibrary.Volume import TypedCoords2Volume

import _Volume

if __name__=='__main__':
	box_size = 60
	resolution = 2.0

	box_size = 60
	resolution = 0.5

	sequence = ['GGAGRRRGGWG']
	angles = torch.zeros(len(sequence), 7, len(sequence[0]), dtype=torch.double)
	angles[0,0,:] = -1.047
	angles[0,1,:] = -0.698
	angles[0,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	c2c = Coords2Center()
	trans = CoordsTranslate()
	c2tc = Coords2TypedCoords()
	tc2v = TypedCoords2Volume(box_size=box_size, resolution=resolution)

	protein, chain_names, res_names, res_nums, atom_names, num_atoms = a2c(angles, sequence)
	center = c2c(protein, num_atoms)
	protein = trans(protein, -center + resolution*box_size/2.0, num_atoms)
	coords, num_atoms_of_type = c2tc(protein, res_names, atom_names, num_atoms)
	print(coords.size(), num_atoms_of_type.size())
	volume = tc2v(coords.to(dtype=torch.float32, device='cuda'), num_atoms_of_type.cuda())
	
	# print(volume.size())
	# for i in range(0, res_names.size(1)):
	# 	print(res_names.data[0,i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0], atom_names.data[0,i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0])

	for i in range(0,11):
		print(num_atoms_of_type.data[0,i])
	
	volume = volume.sum(dim=1).squeeze().cpu()
	print(volume.size(), volume.dtype)
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	_Volume.Volume2Xplor(volume.squeeze(), "TestFig/total_vtest_%d_%.1f.xplor"%(box_size, resolution), resolution)

	