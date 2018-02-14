from __future__ import print_function
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
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PythonInterface import cppPDB2Coords
from PythonInterface import Coords2RMSD
from PythonInterface import Angles2Coords, Angles2Coords_save
from PythonInterface import visSequence, updateAngles

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from PeptideBuilder import Geometry
import PeptideBuilder
import Bio.PDB
from Bio.PDB.Vector import calc_angle


def generateAA(aaName):
	geo = Geometry.geometry(aaName)
	geo.phi=-60
	# geo.phi=0
	geo.psi_im1=-40
	# geo.psi_im1=0
	structure = PeptideBuilder.initialize_res(geo)
	out = Bio.PDB.PDBIO()
	out.set_structure(structure)
	out.save( "example.pdb" )

	return structure[0]['A'][1]

if __name__=='__main__':
	generateAA('L')
	sequence = 'L'
	
	num_atoms = cppPDB2Coords.getSeqNumAtoms(sequence, 0)
	num_angles = 7
	print (num_atoms)
	
	target_coords = Variable(torch.DoubleTensor(3*num_atoms))
	cppPDB2Coords.PDB2Coords("example.pdb", target_coords.data)
	
	angles = Variable(torch.DoubleTensor(num_angles, len(sequence)).zero_(), requires_grad=True)
	angles.data[0,0]=-60.0*np.pi/180.0
	angles.data[1,0]=-56.0*np.pi/180.0
	angles.data[2,0]=280.0*np.pi/180.0
	angles.data[3,0]=132.0*np.pi/180.0
	# angles.data[4,0]=180*np.pi/180.0
	
	loss = Coords2RMSD(num_atoms)
	a2c = Angles2Coords(sequence, num_atoms)
	
	v_num_atoms = Variable(torch.IntTensor(1).fill_(num_atoms))
	
	optimizer = optim.Adam([angles], lr = 0.01)
	for i in range(0,5000):
		coords = a2c(angles)
		rmsd = loss(coords, target_coords, v_num_atoms)
		if np.sqrt(rmsd.data[0])<0.40:
			break
		print (rmsd.data[0], np.sqrt(rmsd.data[0]))
		rmsd.backward()
		angles.grad[0,0]=0.0
		angles.grad[1,0]=0.0
		# angles.grad[2,0]=0.0
		# angles.grad[3,0]=0.0
		optimizer.step()
		
	print(angles.data * 180.0/np.pi)
	Angles2Coords_save(sequence, angles.data, "fitted.pdb")