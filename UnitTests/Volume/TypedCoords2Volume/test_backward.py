import sys
import os
import torch
import matplotlib.pylab as plt
import numpy as np

from TorchProteinLibrary.FullAtomModel import Angles2Coords
from TorchProteinLibrary.FullAtomModel import Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import Coords2CenteredCoords
from TorchProteinLibrary.Volume import TypedCoords2Volume

import _Volume

if __name__=='__main__':

	num_atoms = 10
	atom_coords = []
	atom_types = []
	for i in range(0,num_atoms):
		atom_coords.append(1.0 + np.random.rand(3)*110.0)
		atom_types.append(np.random.randint(low=0, high=11))
	  
	num_atoms_of_type = torch.zeros(1,11, dtype=torch.int)
	offsets = torch.zeros(1,11, dtype=torch.int)
	coords = torch.zeros(1, 3*num_atoms, dtype=torch.double)
	potential = torch.zeros(1,11,120,120,120, dtype=torch.float, device='cuda')
	for i in range(0,120):
		potential[0,:,i,:,:] = float(i)/float(120.0) - 0.5

	for atom_type in range(0,11):
		
		for i, atom in enumerate(atom_types):
			if atom == atom_type:
				num_atoms_of_type[0,atom_type]+=1
		
		if atom_type>0:
			offsets[0, atom_type] = offsets[0, atom_type-1] + num_atoms_of_type[0, atom_type-1]
	
	current_num_atoms_of_type = [0 for i in range(11)]
	for i, r in enumerate(atom_coords):
		index = 3*offsets[0, atom_types[i]] + 3*current_num_atoms_of_type[atom_types[i]]
		coords[0, index + 0 ] = r[0]
		coords[0, index + 1 ] = r[1]
		coords[0, index + 2 ] = r[2]
		current_num_atoms_of_type[atom_types[i]] += 1

	print('Test setting:')
	for i, atom_type in enumerate(atom_types):
		print('Type = ', atom_type, 'Coords = ', atom_coords[i][0], atom_coords[i][1], atom_coords[i][2])
	
	for i in range(0,11):
		print('Type = ', i, 'Num atoms of type = ', num_atoms_of_type[0,i], 'Offset = ', offsets[0,i])

	coords.requires_grad_()
	potential.requires_grad_()
	
	tc2v = TypedCoords2Volume()
	density = tc2v(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())
	E_0 = torch.sum(density*potential)
	E_0.backward()
	grad_an = torch.zeros(coords.grad.size(), dtype=torch.double, device='cpu').copy_(coords.grad.data)

	grad_num = []
	x_1 = torch.zeros(1, 3*num_atoms, dtype=torch.double, device='cpu').requires_grad_()
	dx = 0.01
	for i in range(0,3*num_atoms):
		x_1.data.copy_(coords.data)
		x_1.data[0,i] += dx
		
		density = tc2v(x_1.cuda(), num_atoms_of_type.cuda(), offsets.cuda())
		E_1 = torch.sum(density*potential)
		grad_num.append( (E_1.data - E_0.data)/dx )


	fig = plt.figure()
	plt.plot(grad_num, 'r.-', label = 'num grad')
	plt.plot(grad_an[0,:].numpy(),'bo', label = 'an grad')
	plt.legend()
	plt.savefig('TestFig/test_backward.png')
