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
from TorchProteinLibrary.Volume import TypedCoords2Volume, VolumeRotation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, Coords2Center, getRandomRotation
from TorchProteinLibrary.Utils import ScalarField, ProteinStructure
import time
if __name__=='__main__':
	
	box_size = 30
	resolution = 1.0

	sequence = ['GGGGGGGGGG']
	angles = torch.zeros(len(sequence), 7, len(sequence[0]), dtype=torch.float)
	angles[0,0,:] = 0
	angles[0,1,:] = 0
	angles[0,2:,:] = np.pi
	a2c = Angles2Coords()
	c2cc = Coords2Center()
	c2tc = Coords2TypedCoords()
	tc2v = TypedCoords2Volume(box_size=box_size, resolution=resolution)
	rotate = CoordsRotate()
	translate = CoordsTranslate()

	fig = plt.figure(figsize=(10, 10))
	axis = fig.add_subplot(111, projection='3d')
	
	volume_rotate = VolumeRotation()
	R = getRandomRotation(1).to(dtype=torch.float)
	# R = torch.tensor([[[0,1,0], [-1,0,0], [0,0,1]]], dtype=torch.float)
	# R = torch.tensor([[[0,0,1], [0,1,0], [-1,0,0]]], dtype=torch.float)
	# R = torch.tensor([[[1,0,0], [0,0,1], [0,-1,0]]], dtype=torch.float)
	box_center = torch.tensor([[box_size/2.0 - 0.5, box_size/2.0 - 0.5, box_size/2.0 - 0.5]], dtype=torch.float)
	#Generating protein and initial volume
	coords, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequence)
	
	center = c2cc(coords, num_atoms)
	centered_coords = translate(coords, box_center-center, num_atoms)
	tcoords, num_atoms_of_type = c2tc(centered_coords, resnames, atomnames, num_atoms)
	volume = tc2v(tcoords.cuda(), num_atoms_of_type.cuda())

		
	#Rotating protein
	centered_coords = translate(centered_coords, -box_center, num_atoms)
	rotated_coords = rotate(centered_coords, R, num_atoms)
	rotated_coords = translate(rotated_coords, box_center, num_atoms)
	trotated_coords, num_atoms_of_type = c2tc(rotated_coords, resnames, atomnames, num_atoms)
	
	#Reprojecting protein
	volume_crot = tc2v(trotated_coords.cuda(), num_atoms_of_type.cuda())

	#Rotating volume
	volume_rot = volume_rotate(volume, R.to(dtype=torch.float, device='cuda'))
	
	err = torch.sum(torch.abs(volume_crot - volume_rot))/torch.sum(volume_rot)
	print(err)
	print(volume_rot.size())
	
	field = ScalarField(volume.sum(dim=1))
	field.isosurface(0.5, axis=axis, facecolor='r', alpha=0.4)
	
	field_rot = ScalarField(volume_rot.sum(dim=1))
	field_rot.isosurface(0.5, axis=axis, facecolor='g', alpha=0.4)

	field_rot = ScalarField(volume_crot.sum(dim=1))
	field_rot.isosurface(0.5, axis=axis, facecolor='b', alpha=0.4)
		

	plt.show()
	
	