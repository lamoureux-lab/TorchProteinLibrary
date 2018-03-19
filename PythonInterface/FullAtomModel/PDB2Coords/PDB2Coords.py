# TO DO:
# 1. pdb2sequence
# 2. loading and allocating array
# 3. dealing with missing aa

from Exposed import cppPDB2Coords
import torch

from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one
import numpy as np

def get_sequence(structure):
	sequence = ''
	residues = list(structure.get_residues())
	for n, residue in enumerate(residues):
		try:
			sequence += three_to_one(residue.get_resname())
		except:
			raise Exception("Can't extract sequence")
	return sequence

def pdb2sequence(filenames):
	p = PDBParser(PERMISSIVE=1)
	sequences = []
	for filename in filenames:
		structure = p.get_structure('X', filename)
		sequences.append(get_sequence(structure))
	return sequences

def convertStringList(stringList):
	'''Converts list of strings to 0-terminated byte tensor'''
	maxlen = 0
	for string in stringList:
		string += '\0'
		if len(string)>maxlen:
			maxlen = len(string)
	ar = np.zeros( (len(stringList), maxlen), dtype=np.uint8)
	
	for i,string in enumerate(stringList):
		npstring = np.fromstring(string, dtype=np.uint8)
		ar[i,:npstring.shape[0]] = npstring
	
	return torch.from_numpy(ar)

def convertString(string):
	'''Converts a string to 0-terminated byte tensor'''  
	return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))

class PDB2Coords:
	def __init__(self, add_term=False):
		self.add_term = add_term

	def __call__(self, filenames):
		self.filenamesTensor = convertStringList(filenames)
		self.num_atoms = []
		self.sequences = pdb2sequence(filenames)
		self.seqTensor = convertStringList(self.sequences)
		for seq in self.sequences:
			self.num_atoms.append(cppPDB2Coords.getSeqNumAtoms(seq, self.add_term))
		
		max_num_atoms = max(self.num_atoms)
		batch_size = len(self.num_atoms)

		output_coords_cpu = torch.DoubleTensor(batch_size, max_num_atoms*3)
		output_resnames_cpu = torch.ByteTensor(batch_size, max_num_atoms, 4).contiguous()
		output_atomnames_cpu = torch.ByteTensor(batch_size, max_num_atoms, 4).contiguous()

		cppPDB2Coords.PDB2Coords(self.filenamesTensor, output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, self.add_term)

		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu
		
