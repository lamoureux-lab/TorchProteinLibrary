import sys
import os
import torch
import numpy as np

import TorchProteinLibrary
from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import Coords2CenteredCoords
from TorchProteinLibrary.Volume import TypedCoords2Volume, SelectVolume

if __name__=='__main__':
	sequence = ['GGAGRRRGGWG', 'GGAGRRR']
	angles = torch.zeros(len(sequence), 7, len(sequence[0]), dtype=torch.double)
	angles[0,0,:] = -1.047
	angles[0,1,:] = -0.698
	angles[0,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	c2cc = Coords2CenteredCoords()
	c2tc = Coords2TypedCoords()
	tc2v = TypedCoords2Volume()
	
	protein, res_names, atom_names, num_atoms = a2c(angles, sequence)
	protein = c2cc(protein, num_atoms)
	coords, num_atoms_of_type, offsets = c2tc(protein, res_names, atom_names, num_atoms)
	volume = tc2v(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())

	sv = SelectVolume()
	features = sv(volume, coords.float().cuda(), num_atoms.cuda())
	print(features.sum())
	batch_idx = 0
	feature_idx = 1
	res = 1.0
	num_max_atoms = int(coords.size(1)/3)
	err = 0.0
	for i in range(0,num_max_atoms):
		x = int(np.floor(coords[batch_idx,3*i]/res))
		y = int(np.floor(coords[batch_idx,3*i+1]/res))
		z = int(np.floor(coords[batch_idx,3*i+2]/res))
		err += np.abs(features[batch_idx, feature_idx, i] - volume[batch_idx, feature_idx, x, y, z])
	
	print ('Error = ', err/num_max_atoms)