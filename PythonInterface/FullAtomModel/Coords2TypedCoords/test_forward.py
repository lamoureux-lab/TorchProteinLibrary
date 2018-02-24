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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Angles2Coords.Angles2Coords import Angles2Coords
from Coords2TypedCoords import Coords2TypedCoords

if __name__=='__main__':

	sequence = 'GGGGGG'
	angles = Variable(torch.DoubleTensor(7,len(sequence)).zero_())
	angles[0,:] = -1.047
	angles[1,:] = -0.698
	angles[2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords(sequence)
	protein, res_names, atom_names = a2c(angles)
	
	for i in range(0, res_names.size(0)):
		print res_names.data[i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0], atom_names.data[i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0]

	c2tc = Coords2TypedCoords(num_atoms= atom_names.size(0))
	coords, num_atoms_of_type, offsets = c2tc(protein,res_names,atom_names)
	for i in range(0,11):
		print num_atoms_of_type.data[i], offsets.data[i]

	

	