#!/usr/bin/python

'''
Simple example script demonstrating how to use the PeptideBuilder library.

The script generates a peptide consisting of six arginines in alpha-helix
conformation, and it stores the peptide under the name "example.pdb".
'''

from __future__ import print_function
from PeptideBuilder import Geometry
import PeptideBuilder
import Bio.PDB
from Bio.PDB.Vector import calc_angle


def generateAA(aaName):
	geo = Geometry.geometry(aaName)
	geo.phi=-60
	geo.psi_im1=-40
	structure = PeptideBuilder.initialize_res(geo)
	out = Bio.PDB.PDBIO()
	out.set_structure(structure)
	out.save( "example.pdb" )

	return structure[0]['A'][1]

def getCG1_CB_CG2_angle():
	residue = generateAA('V')
	v1 = residue['CG1'].get_vector()
	v2 = residue['CG2'].get_vector()
	v3 = residue['CB'].get_vector()
	CG1_CB_CG2_angle = calc_angle(v1,v3,v2)
	print(CG1_CB_CG2_angle)

if __name__=='__main__':
	generateAA('I')