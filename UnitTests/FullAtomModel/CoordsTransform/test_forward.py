import sys
import os
import torch
import numpy as np
from TorchProteinLibrary.FullAtomModel.CoordsTransform import CoordsTranslate, getRandomTranslation, getBBox, CoordsRotate, getRandomRotation, getRotation, getSO3Samples
from TorchProteinLibrary.FullAtomModel import Angles2Coords, Coords2TypedCoords

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sea
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pylab as plt

def test_translation(coords, num_atoms):
	translate = CoordsTranslate()
	a,b = getBBox(coords, num_atoms)
	center = (a+b)*0.5
	print (center)

	centered_coords = translate(coords, -center, num_atoms)
	a,b = getBBox(centered_coords, num_atoms)
	center = (a+b)*0.5
	
	print(center)
	

def test_random_rotation(num_rotations):
	batch_size = num_rotations
	R = getRandomRotation(batch_size)
		
	num_atoms = torch.ones(R.size(0), dtype=torch.int)
	coords = torch.zeros(R.size(0),3, dtype=torch.double)
	coords[:,0]=1.0
	coords[:,1]=0.0
	coords[:,2]=0.0

	rotate = CoordsRotate()
	rotated = rotate(coords, R, num_atoms)
	
	return rotated

def test_rotation(angle_inc = 0.1):
	u = []
	for alpha in np.arange(0.0, 1.0, angle_inc):
		for beta in np.arange(0.0, 1.0, angle_inc):
			for gamma in np.arange(0.0, 1.0, angle_inc):
				u.append([alpha, beta, gamma])

	u = np.array(u)
	u = torch.from_numpy(u)
	u = u.to(dtype=torch.double)

	R = getRotation(u)
	rotate = CoordsRotate()
	
	num_atoms = torch.ones(u.size(0), dtype=torch.int)
	coords = torch.zeros(u.size(0),3, dtype=torch.double)
	coords[:,0]=1.0
	coords[:,1]=0.0
	coords[:,2]=0.0

	rotated = rotate(coords, R, num_atoms)
	return rotated

def plot_coords(coords, filename, plot_name='Rotation test'):
	if not os.path.exists("TestFig"):
		os.mkdir("TestFig")

	min_xyz = -1.5
	max_xyz = 1.5
	coords = coords.numpy()
	sx, sy, sz = coords[:,0], coords[:,1], coords[:,2]
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.plot(sx, sy, sz, '.', label = plot_name)
	ax.set_xlim(min_xyz,max_xyz)
	ax.set_ylim(min_xyz,max_xyz)
	ax.set_zlim(min_xyz,max_xyz)
	ax.legend()
	plt.show()
	plt.savefig('TestFig/%s'%filename)


if __name__=='__main__':

	sequences = ['GGGGGG', 'GGAARRRRRRRRR']
	angles = torch.zeros(2, 7,len(sequences[1]), dtype=torch.double)
	angles[:,0,:] = -1.047
	angles[:,1,:] = -0.698
	angles[:,2:,:] = 110.4*np.pi/180.0
	a2c = Angles2Coords()
	protein, res_names, atom_names, num_atoms = a2c(angles, sequences)
	
	test_translation(protein, num_atoms)
	
	rotated = test_random_rotation(1000)
	plot_coords(rotated, "random_rotation.png")
	
	rotated = test_rotation()
	plot_coords(rotated, "uniform_rotation.png")

		
	