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

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from PeptideBuilder import Geometry
import PeptideBuilder
import Bio.PDB
from Bio.PDB.Vector import calc_angle


def generateAA(aaName):
	geo = Geometry.geometry(aaName)
	# geo.phi=-60
	geo.phi=0
	# geo.psi_im1=-40
	geo.psi_im1=0
	structure = PeptideBuilder.initialize_res(geo)
	out = Bio.PDB.PDBIO()
	out.set_structure(structure)
	out.save( "example.pdb" )

	return structure[0]['A'][1]

if __name__=='__main__':
	generateAA('A')
	sequence = 'A'
	num_atoms = cppPDB2Coords.getSeqNumAtoms(sequence)
	num_angles = 7
	print (num_atoms)
	
	target_coords = Variable(torch.DoubleTensor(3*num_atoms))
	cppPDB2Coords.PDB2Coords("example.pdb", target_coords.data)
	
	angles = Variable(torch.DoubleTensor(num_angles, len(sequence)).zero_(), requires_grad=True)
	angles.data[0,0]=np.pi
	
	loss = Coords2RMSD(num_atoms)
	a2c = Angles2Coords(sequence, num_atoms)
	
	v_num_atoms = Variable(torch.IntTensor(1).fill_(num_atoms))
	
	optimizer = optim.Adam([angles], lr = 0.0005)
	for i in range(0,10000):
		coords = a2c(angles)
		rmsd = loss(coords, target_coords, v_num_atoms)
		rmsd_real = np.sqrt(rmsd.data[0])
		print (rmsd_real)
		rmsd.backward()
		optimizer.step()

	Angles2Coords_save(sequence, angles.data, "fitted.pdb")