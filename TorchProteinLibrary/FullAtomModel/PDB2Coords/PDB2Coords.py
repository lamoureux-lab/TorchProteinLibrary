# TO DO:
# 1. pdb2sequence
# 2. loading and allocating array
# 3. dealing with missing aa

import torch
from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import os 
import sys
import _FullAtomModel

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
	p = PDBParser(PERMISSIVE=1, QUIET=True)
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

def string2tensor(string):
	'''Converts a string to 0-terminated byte tensor'''  
	return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))

def tensor2string(tensor):
	return tensor.numpy().tostring().split(b'\x00')[0].decode("utf-8", "strict")

class PDB2CoordsOrdered:
					
	def __call__(self, filenames):
		
		filenamesTensor = convertStringList(filenames)
		max_num_atoms = 1
		batch_size = len(self.filenames)

		num_atoms = torch.zeros(batch_size, dtype=torch.int)
		output_coords_cpu = torch.zeros(batch_size, max_num_atoms*3, dtype=torch.double)
		output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
		output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
		mask = torch.zeros(batch_size, max_num_atoms, dtype=torch.uint8)

		_FullAtomModel.PDB2CoordsOrdered(filenamesTensor, sequences, output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, num_atoms, mask)
	
		return output_coords_cpu, mask, output_resnames_cpu, output_atomnames_cpu, num_atoms

class PDB2CoordsUnordered:
					
	def __call__(self, filenames):
		
		filenamesTensor = convertStringList(filenames)
		batch_size = len(filenames)
		num_atoms = torch.zeros(batch_size, dtype=torch.int)
		output_coords_cpu = torch.zeros(batch_size, 1, dtype=torch.double)
		output_chainnames_cpu = torch.zeros(batch_size, 1, 1, dtype=torch.uint8)
		output_resnames_cpu = torch.zeros(batch_size, 1, 1, dtype=torch.uint8)
		output_resnums_cpu = torch.zeros(batch_size, 1, dtype=torch.int)
		output_atomnames_cpu = torch.zeros(batch_size, 1, 1, dtype=torch.uint8)

		_FullAtomModel.PDB2CoordsUnordered(filenamesTensor, output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms)
	
		return output_coords_cpu, output_chainnames_cpu, output_resnames_cpu, output_resnums_cpu, output_atomnames_cpu, num_atoms

def writePDB(filename, coords, chainnames, resnames, resnums, atomnames, num_atoms, bfactors=None, add_model=True, rewrite=True):
	batch_size = coords.size(0)
	last_model_num = 0
	last_atom_num = 0
		
	if os.path.exists(filename):
		if rewrite:
			os.remove(filename)
		else:	
			with open(filename, 'r') as fin:
				for line in fin:
					if line.find('MODEL') != -1:
						sline = line.split()
						model_num = int(sline[1])
						if last_model_num<model_num:
							last_model_num = model_num
					if line.find('ATOM') != -1:
						atom_num = int(line[6:11])
						if last_atom_num<atom_num:
							last_atom_num = atom_num

	with open(filename, 'a') as fout:
		for i in range(batch_size):
			if add_model:
				fout.write("MODEL %d\n"%(i+last_model_num))
			
			for j in range(num_atoms[i].item()):
				chain_name = tensor2string(chainnames[i,j,:])
				if len(chain_name) == 0:
					chain_name = "A"
				# print(chain_name)
				atom_name = tensor2string(atomnames[i,j,:])
				res_name = tensor2string(resnames[i,j,:])
				res_num = resnums[i,j].item()
				x = coords[i, 3*j].item()
				y = coords[i, 3*j+1].item()
				z = coords[i, 3*j+2].item()
				if bfactors is None:
					fout.write("ATOM  %5d  %-3s %3s %c%4d    %8.3f%8.3f%8.3f\n"%(j + last_atom_num + 1, atom_name, res_name, chain_name[0], res_num, x, y, z))
				else:
					bfactor = bfactors[i, j]
					fout.write("ATOM  %5d  %-3s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(j + last_atom_num + 1, atom_name, res_name, chain_name[0], res_num, x, y, z, 1.0, bfactor))
			
			if add_model:
				fout.write("ENDMDL\n")


