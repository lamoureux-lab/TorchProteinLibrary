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

from TorchProteinLibrary.RMSD import Coords2RMSD
from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import Coords2CenteredCoords
from TorchProteinLibrary.Volume import VolumeRMSD
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsRotate, getRandomRotation, CoordsTranslate, getBBox
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords

import _Volume




if __name__=='__main__':
	
	box_size = 80
	resolution = 1.25
	ix = 0
	iy = 159
	iz = 0

	sequence = ['GGAGRRRGGWG']
	angles = torch.zeros(len(sequence), 7, len(sequence[0]), dtype=torch.double)
	angles[0,0,:] = -1.047
	angles[0,1,:] = -0.698
	angles[0,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	rotate = CoordsRotate()
	translate = CoordsTranslate()
	rmsd = Coords2RMSD()
	R1 = getRandomRotation( len(sequence) )
	R0 = torch.zeros(1,3,3, dtype=torch.double, device='cpu')
	R0[0,0,0]=1.0
	R0[0,1,1]=1.0
	R0[0,2,2]=1.0
	T0 = torch.zeros(1,3, dtype=torch.double, device='cpu')
	R1 = R0
	#Generating protein
	protein, res_names, atom_names, num_atoms = a2c(angles, sequence)
		
	rmsd_vol = VolumeRMSD(protein, num_atoms, R0, R1, T0, resolution, box_size*2)

	#Rotating and translating protein
	T1 = torch.zeros(1, 3, dtype=torch.double, device='cpu')
	T1[0,0] = ix * resolution
	T1[0,1] = iy * resolution
	T1[0,2] = iz * resolution
	if ix>=box_size:
		T1[0,0] = -(2*box_size - ix)*resolution
	if iy>=box_size:
		T1[0,1] = -(2*box_size - iy)*resolution
	if iz>=box_size:
		T1[0,2] = -(2*box_size - iz)*resolution
	
	rotated_coords = rotate(protein, R1, num_atoms)
	trans_rotated_coords = translate(rotated_coords, T1, num_atoms)

	rmsd2 = protein[0, :3*num_atoms[0].item()] - trans_rotated_coords[0,:3*num_atoms[0].item()]
	rmsd2 = torch.sum(rmsd2*rmsd2)/num_atoms[0].item()
	rmsd = torch.sqrt(rmsd2)

	print('VolumeRMSD: ', rmsd_vol[0, ix, iy, iz].item())
	print('Coords2RMSD: ', rmsd)