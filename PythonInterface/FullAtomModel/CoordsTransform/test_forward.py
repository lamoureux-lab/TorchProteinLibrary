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
from CoordsTransform import CoordsTranslate, getRandomTranslation, getBBox
from CoordsTransform import CoordsRotate, getRandomRotation



def test_translation(coords, num_atoms):
	translate = CoordsTranslate()
	a,b = getBBox(coords, num_atoms)
	center = (a+b)*0.5

	centered_coords = translate(coords, -center, num_atoms)
	a,b = getBBox(centered_coords, num_atoms)
	center = (a+b)*0.5
	
	print(center)
	

def test_rotation(coords, num_atoms):
	batch_size = num_atoms.size(0)
	R = getRandomRotation(batch_size)
	rotate = CoordsRotate()
	rotated = rotate(coords, R, num_atoms)
	
	print(rotated)
	

if __name__=='__main__':

	sequences = ['GGGGGG', 'GGAARRRRRRRRR']
	angles = torch.zeros(2, 7,len(sequences[1]), dtype=torch.double)
	angles[:,0,:] = -1.047
	angles[:,1,:] = -0.698
	angles[:,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	protein, res_names, atom_names, num_atoms = a2c(angles, sequences)
	
	test_translation(protein, num_atoms)
	test_rotation(protein, num_atoms)
		
	