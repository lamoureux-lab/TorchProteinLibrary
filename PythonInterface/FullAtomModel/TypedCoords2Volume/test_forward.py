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
from Coords2TypedCoords.Coords2TypedCoords import Coords2TypedCoords
from Coords2CenteredCoords import Coords2CenteredCoords

from TypedCoords2Volume import TypedCoords2Volume

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import Utils

if __name__=='__main__':

	sequence = ['GGGGGG']
	angles = Variable(torch.DoubleTensor(len(sequence), 7,len(sequence[0])).zero_())
	angles[0,0,:] = -1.047
	angles[0,1,:] = -0.698
	angles[0,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	c2cc = Coords2CenteredCoords()
	c2tc = Coords2TypedCoords()
	tc2v = TypedCoords2Volume()
	protein, res_names, atom_names = a2c(angles, sequence)
	protein = c2cc(protein)
	coords, num_atoms_of_type, offsets = c2tc(protein,res_names,atom_names, a2c.num_atoms)
	volume = tc2v(coords, num_atoms_of_type, offsets)

	for i in range(0, res_names.size(1)):
		print res_names.data[0,i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0], atom_names.data[0,i,:].numpy().astype(dtype=np.uint8).tostring().split('\0')[0]

	for i in range(0,11):
		print num_atoms_of_type.data[0,i], offsets.data[0,i]
	
	volume = volume.cpu().data
	for i in xrange(volume.size(1)):
		Utils.SaveVolumeXPlor(volume[0,i,:,:,:], "TestFig/vtest%d.xplor"%(i))

	volume = volume.sum(dim=1)
	Utils.SaveVolumeXPlor(volume[0,:,:,:], "TestFig/total_vtest.xplor")

	