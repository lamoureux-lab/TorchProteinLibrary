import os
import sys
import argparse
import torch
from TorchProteinLibrary import FullAtomModel
from utils import transform, bytes2string
import numpy as np
from rotamers import getAngles, generateAA
import unittest
from Bio.PDB.Polypeptide import aa1

class TestAngles2CoordsForward(unittest.TestCase):

	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()

	def _generate_reference(self, aa):
		self.reference = {}
		structure = generateAA(aa)
		structure = transform(structure)
		self.angles = getAngles(structure)
		for atom in structure.get_atoms():
			self.reference[atom.get_name()] = np.array(atom.get_coord())
	
	def _generate_sample(self, aa):
		sequences = [aa]
		protein, res_names, atom_names, num_atoms = self.a2c(self.angles, sequences)
		self.sample = {}
		for i in range(atom_names.size(1)):
			self.sample[bytes2string(atom_names[0,i,:])] = protein.data[0,3*i : 3*i + 3].numpy()

	def _measure_rmsd(self):
		N = len(self.reference.keys())
		RMSD = 0.0
		for atom_name in self.reference.keys():
			norm = np.linalg.norm(self.reference[atom_name] - self.sample[atom_name])
			RMSD += norm*norm
		RMSD /= float(N*(N-1))
		return np.sqrt(RMSD)

	def runTest(self):
		print('Amino-acid\tRMSD')
		for aa in aa1:
			self._generate_reference(aa)
			self._generate_sample(aa)
			rmsd = self._measure_rmsd()
			print("%s\t\t%f"%(aa, rmsd))
			self.assertLess(rmsd, 1.0)
	
if __name__=='__main__':
	unittest.main()
	


