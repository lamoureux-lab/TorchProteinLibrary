from __future__ import print_function
import sys
import os

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import Bio.PDB
from Bio.PDB import calc_angle, rotaxis, Vector, calc_dihedral
from math import *
import numpy as np

import PeptideBuilder
from PeptideBuilder import Geometry

import torch

def generateAA(aaName):
	geo = Geometry.geometry(aaName)
	structure = PeptideBuilder.initialize_res(geo)
	return structure

def generateSeq(seq):
	structure = None	
	for aa in seq:
		geo = Geometry.geometry(aa)
		if structure is None:
			structure = PeptideBuilder.initialize_res(geo)
		else:
			structure = PeptideBuilder.add_residue(structure, geo)
		
	return structure

def getAngles(structure):
	residues = list(structure.get_residues())
	angles = torch.zeros(1, 8, len(residues), dtype=torch.double, device='cpu')
	phi, psi = getBackbone(residues)
	for i, residue in enumerate(residues):
		angles[0, 0, i] = phi[i]
		angles[0, 1, i] = psi[i]
		angles[0, 2, i] = np.pi
		xis = getRotamer(residue)
		for j, xi in enumerate(xis):
			angles[0, 3+j, i] = xis[j]
	return angles


def getBackbone(residues):
	phi = [0.0]
	psi = []
	for i, res_i in enumerate(residues):
		N_i = res_i["N"].get_vector()
		CA_i = res_i["CA"].get_vector()
		C_i = res_i["C"].get_vector()
		if i>0:
			res_im1 = residues[i-1]
			C_im1 = res_im1["C"].get_vector()
			phi.append(calc_dihedral(C_im1, N_i, CA_i, C_i))
		if i<(len(residues)-1):
			res_ip1 = residues[i+1]
			N_ip1 = res_ip1["N"].get_vector()
			psi.append(calc_dihedral(N_i, CA_i, C_i, N_ip1))
	psi.append(0.0)
	return phi, psi

def getRotamer(residue):
	if residue.get_resname()=='GLY':
		return []
	if residue.get_resname()=='ALA':
		return []
	if residue.get_resname()=='CYS':
		return getCysRot(residue)
	if residue.get_resname()=='SER':
		return getSerRot(residue)
	if residue.get_resname()=='VAL':
		return getValRot(residue)
	if residue.get_resname()=='ILE':
		return getIleRot(residue)
	if residue.get_resname()=='LEU':
		return getLeuRot(residue)
	if residue.get_resname()=='THR':
		return getThrRot(residue)
	if residue.get_resname()=='ARG':
		return getArgRot(residue)
	if residue.get_resname()=='LYS':
		return getLysRot(residue)
	if residue.get_resname()=='ASP':
		return getAspRot(residue)
	if residue.get_resname()=='ASN':
		return getAsnRot(residue)
	if residue.get_resname()=='GLU':
		return getGluRot(residue)
	if residue.get_resname()=='GLN':
		return getGlnRot(residue)
	if residue.get_resname()=='MET':
		return getMetRot(residue)
	if residue.get_resname()=='HIS':
		return getHisRot(residue)
	if residue.get_resname()=='PRO':
		return getProRot(residue)
	if residue.get_resname()=='PHE':
		return getPheRot(residue)
	if residue.get_resname()=='TYR':
		return getTyrRot(residue)
	if residue.get_resname()=='TRP':
		return getTrpRot(residue)

def getCysRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	SG = residue["SG"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, SG) + np.pi
	return [xi1]

def getSerRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	OG = residue["OG"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, OG) + np.pi
	return [xi1]

def getValRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG1 = residue["CG1"].get_vector()
	CG2 = residue["CG2"].get_vector()
	xi1 = (calc_dihedral(N, CA, CB, CG1) + calc_dihedral(N, CA, CB, CG2))/2.0 + np.pi
	return [xi1]

def getIleRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG1 = residue["CG1"].get_vector()
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG1)
	xi2 = calc_dihedral(CA, CB, CG1, CD1)+np.pi
	return [xi1, xi2]

def getLeuRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()
	CD1 = residue["CD1"].get_vector()
	CD2 = residue["CD2"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = (calc_dihedral(CA, CB, CG, CD1) + calc_dihedral(CA, CB, CG, CD2))/2.0
	return [xi1, xi2]

def getThrRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	OG1 = residue["OG1"].get_vector()	
	CG2 = residue["CG2"].get_vector()
	xi1 = (calc_dihedral(N, CA, CB, OG1) + calc_dihedral(N, CA, CB, CG2))/2.0
	return [xi1]

def getArgRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	NE = residue["NE"].get_vector()
	CZ = residue["CZ"].get_vector()
	NH1 = residue["NH1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, NE)
	xi4 = calc_dihedral(CG, CD, NE, CZ)
	xi5 = calc_dihedral(CD, NE, CZ, NH1)
	return [xi1, xi2, xi3, xi4, xi5]

def getLysRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	CE = residue["CE"].get_vector()
	NZ = residue["NZ"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, CE)
	xi4 = calc_dihedral(CG, CD, CE, NZ)
	return [xi1, xi2, xi3, xi4]

def getAspRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	OD1 = residue["OD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, OD1)
	return [xi1, xi2]

def getAsnRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	OD1 = residue["OD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, OD1)
	return [xi1, xi2]

def getGluRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	OE1 = residue["OE1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, OE1)
	return [xi1, xi2, xi3]

def getGlnRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	OE1 = residue["OE1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, OE1)
	return [xi1, xi2, xi3]

def getMetRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	SD = residue["SD"].get_vector()
	CE = residue["CE"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, SD)
	xi3 = calc_dihedral(CB, CG, SD, CE)
	return [xi1, xi2, xi3]

def getHisRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	ND1 = residue["ND1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, ND1)+np.pi/2.0
	return [xi1, xi2]

def getProRot(residue):
	return []

def getPheRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)+np.pi/2.0
	return [xi1, xi2]

def getTyrRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)+np.pi/2.0
	return [xi1, xi2]

def getTrpRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)-np.pi/2.0
	return [xi1, xi2]




if __name__=='__main__':
	structure = generateSeq('CCCCC')
	print (getAngles(structure))
	# for model in structure:
	# 	for chain in model:
	# 		for residue in chain:
	# 			for atom in residue:
	# 				print (atom)
	# 			print (getRotamer(residue))
	# print (getBackbone(list(structure.get_residues())))