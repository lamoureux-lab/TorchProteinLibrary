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
from Bio.PDB.Vector import calc_angle, rotaxis, Vector
from math import *

from rotamers import getRotamer, getBackbone, getAngles, generateSeq

def generateAA(aaName):
	geo = Geometry.geometry(aaName)
	geo.phi=0
	geo.psi_im1=0
	structure = PeptideBuilder.initialize_res(geo)
	
	tx = -np.pi/2.0
	Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
	for atom in structure.get_atoms():
		atom.transform(Rx, np.array([0,0,0]))

	nAtom = list(structure.get_atoms())[0]
	nV = nAtom.get_coord()
	I = np.identity(3)
	for atom in structure.get_atoms():
		atom.transform(I, -nV)

	R = rotaxis(np.pi, list(structure.get_atoms())[1].get_vector())
	for atom in structure.get_atoms():
		atom.transform(R, np.array([0,0,0]))

	print(list(structure.get_atoms())[1].get_coord(), list(structure.get_atoms())[1])
	
	out = Bio.PDB.PDBIO()
	out.set_structure(structure)
	out.save( "example.pdb" )

	return structure[0]['A'][1]

def transform(structure):
	tx = -np.pi/2.0
	Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
	for atom in structure.get_atoms():
		atom.transform(Rx, np.array([0,0,0]))

	nAtom = list(structure.get_atoms())[0]
	nV = nAtom.get_coord()
	I = np.identity(3)
	for atom in structure.get_atoms():
		atom.transform(I, -nV)

	R = rotaxis(np.pi, list(structure.get_atoms())[1].get_vector())
	for atom in structure.get_atoms():
		atom.transform(R, np.array([0,0,0]))

	return structure

def save(structure):
	out = Bio.PDB.PDBIO()
	out.set_structure(structure)
	out.save( "example.pdb" )

if __name__=='__main__':
	# residue = generateAA('S')
	sequence = 'SCH'
	structure = generateSeq(sequence)
	structure = transform(structure)
	save(structure)
	
	num_atoms = cppPDB2Coords.getSeqNumAtoms(sequence, 0)
	num_angles = 7
	
	target_coords = Variable(torch.DoubleTensor(3*num_atoms))
	cppPDB2Coords.PDB2Coords("example.pdb", target_coords.data)
	
	angles = getAngles(structure)
	angles = Variable(angles, requires_grad=True)
	
	loss = Coords2RMSD(num_atoms)
	a2c = Angles2Coords(sequence, num_atoms)
	
	v_num_atoms = Variable(torch.IntTensor(1).fill_(num_atoms))
	
	optimizer = optim.Adam([angles], lr = 0.0001)
	y = []
	for i in range(0,10000):
		coords = a2c(angles)
		rmsd = loss(coords, target_coords, v_num_atoms)
		if np.sqrt(rmsd.data[0])<0.10:
			break
		print (rmsd.data[0], np.sqrt(rmsd.data[0]))
		y.append(np.sqrt(rmsd.data[0]))
		rmsd.backward()
		
		optimizer.step()
		
	print(angles.data * 180.0/np.pi)
	Angles2Coords_save(sequence, angles.data, "fitted.pdb", False)
	plt.plot(y)
	plt.show()