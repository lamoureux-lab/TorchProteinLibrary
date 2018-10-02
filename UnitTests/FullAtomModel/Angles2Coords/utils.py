from PeptideBuilder import Geometry
import PeptideBuilder
import Bio.PDB
from Bio.PDB import calc_angle, rotaxis, Vector
from math import *
import numpy as np

def bytes2string(tbt_array):
	return tbt_array.numpy().astype(dtype=np.uint8).tostring().split(b'\00')[0].decode("utf-8") 


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

	# print(list(structure.get_atoms())[1].get_coord(), list(structure.get_atoms())[1])
	
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