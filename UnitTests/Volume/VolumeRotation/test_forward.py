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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import Coords2CenteredCoords
from TorchProteinLibrary.Volume import TypedCoords2Volume, VolumeRotation
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsRotate, getRandomRotation, CoordsTranslate, getBBox
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords

import _Volume




if __name__=='__main__':
	
	box_size = 80
	resolution = 1.25

	sequence = ['GGAGRRRGGWG']
	angles = torch.zeros(len(sequence), 7, len(sequence[0]), dtype=torch.double)
	angles[0,0,:] = -1.047
	angles[0,1,:] = -0.698
	angles[0,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	c2cc = Coords2CenteredCoords(rotate=False, translate=False, box_size=box_size, resolution=resolution)
	c2tc = Coords2TypedCoords()
	tc2v = TypedCoords2Volume(box_size=box_size, resolution=resolution)
	rotate = CoordsRotate()
	translate = CoordsTranslate()
	R = getRandomRotation( len(sequence) )
	volume_rotate = VolumeRotation(mode='bilinear')

	
	#Generating protein and initial volume
	protein, res_names, atom_names, num_atoms = a2c(angles, sequence)
	protein = c2cc(protein, num_atoms)
	coords, num_atoms_of_type, offsets = c2tc(protein, res_names, atom_names, num_atoms)
	volume = tc2v(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())

	#Rotating protein
	center = torch.zeros(len(sequence), 3, dtype=torch.double, device='cpu').fill_((box_size - 0.5)*resolution/2.0)
	centered_coords = translate(coords, -center, num_atoms)
	rotated_centered_coords = rotate(centered_coords, R, num_atoms)
	rotated_coords = translate(rotated_centered_coords, center, num_atoms)
	#Reprojecting protein
	volume_protRot = tc2v(rotated_coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())

	#Rotating volume	
	volume_rot = volume_rotate(volume, R.to(dtype=torch.float, device='cuda'))
	
	err = torch.sum(torch.abs(volume_protRot - volume_rot))/torch.sum(volume_rot)
	print(err)
	
	volume = volume.sum(dim=1).squeeze().cpu()
	volume_protRot = volume_protRot.sum(dim=1).squeeze().cpu()
	volume_rot = volume_rot.sum(dim=1).squeeze().cpu()


	
	if not os.path.exists('TestFig'):
		os.mkdir('TestFig')
	_Volume.Volume2Xplor(volume_protRot.squeeze(), "TestFig/total_vPRot_%d_%.1f.xplor"%(box_size, resolution), resolution)
	_Volume.Volume2Xplor(volume_rot.squeeze(), "TestFig/total_vRot_%d_%.1f.xplor"%(box_size, resolution), resolution)
	_Volume.Volume2Xplor(volume.squeeze(), "TestFig/total_v_%d_%.1f.xplor"%(box_size, resolution), resolution)