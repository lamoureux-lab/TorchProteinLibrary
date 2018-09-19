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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from tqdm import tqdm
import Bio.PDB
from Bio.PDB.Polypeptide import is_aa,d3_to_index,dindex_to_1,standard_aa_names 
from Bio import SeqIO
from Datasets.GenerateDataset.conf import PFAM_DATASET_SEQUENCES, CAMEO_OLD_DATASET_SEQUENCES, MEMPROT_DATASET_SEQUENCES, \
				PDB25_DATASET_SEQUENCES_NOHOMO, DATASET_TEST_FINAL_LOCAL_DIR, DATASET_TRAIN_FINAL_LOCAL_DIR
from Datasets.GenerateDataset.conf import logging
from Datasets.GenerateDataset.util import choose_atom, get_fasta_seq, get_normal_alignments

from Layers import Coords2RMSD, Angles2CoordsDihedral


def extract_residues_with_CA(filepath, chain_id):
	"""Extracts list of residues from a chain that have backbone atoms and belong to the protein. Uses Biopython"""
	structure = Bio.PDB.PDBParser(QUIET=True).get_structure("x", filepath)
	model = structure[0]
	chain = model[chain_id]
	protein_residues = []

	for residue in chain:
		rname = residue.get_resname()
		if rname.find('H_')!=-1:
			continue
		if not is_aa(residue,standard=False):
			continue
		if not residue.has_id('CA'):
			print 'Incomplete residue'
			continue
		if choose_atom(residue) is None:
			print 'Can not chose ref atom'
			continue
		protein_residues.append(residue)
	return protein_residues

def calc_contact_matrix_CA(filepath, chain_id, threshold2):
	"""Returns a matrix of C-alpha distances between two chains. Uses Biopython"""
	protein_residues = extract_residues_with_CA(filepath, chain_id)
	
	matrix = np.zeros((len(protein_residues), len(protein_residues)))

	for row, residue_one in enumerate(protein_residues):
		r1 = choose_atom(residue_one)
		for col, residue_two in enumerate(protein_residues):
			r2 = choose_atom(residue_two)
			diff_vector  = r1.coord - r2.coord
			if np.sum(diff_vector * diff_vector)<threshold2:
				matrix[row, col] = 1.0
			else:
				matrix[row, col] = 0.0
	return matrix


def extract_CA(filename, chain_id):
	print filename
	protein_residues = extract_residues_with_CA(filename, chain_id)
	#shifting vectors to the frame origin
	r_0 = protein_residues[0]['CA'].get_vector()
	vectors = []
	for i in range(0,len(protein_residues)):
		residue_i = protein_residues[i]
		vectors.append(residue_i['CA'].get_vector() - r_0)
	targets = torch.zeros(3*len(vectors))
	k=0
	for i in range(0,len(vectors)):
		targets[k] = vectors[i][0]
		k+=1
		targets[k] = vectors[i][1]
		k+=1
		targets[k] = vectors[i][2]
		k+=1
	return targets, len(vectors)

def get_structure(coords):
	x = []
	y = []
	z = []
	for i in range(0, coords.size()[0]/3, 3):
		x.append(coords.data[i])
		y.append(coords.data[i+1])
		z.append(coords.data[i+2])
	return x,y,z
	


if __name__=='__main__':

	dataset_sequences = PFAM_DATASET_SEQUENCES
	final_dataset_path = DATASET_TEST_FINAL_LOCAL_DIR
	dataset_structures = DATASET_TEST_FINAL_LOCAL_DIR
	data = get_fasta_seq(dataset_sequences)
	normalData = get_normal_alignments(data, final_dataset_path)
	query_id = normalData[0][0]
	seq = normalData[0][1]
	filename = os.path.join(dataset_structures,query_id+'.pdb')

	target_coords, L = extract_CA(filename, query_id[5:])
	target_coords = Variable(target_coords).cuda()
	x = Variable(torch.randn(2,L-1).cuda(), requires_grad=True)
	length = Variable(torch.IntTensor(1).fill_(L-1))

	a2c = Angles2CoordsDihedral(L-1)
	c2RMSD = Coords2RMSD(L-1)

	optimizer = optim.Adam([x], lr = 0.01)
	trace = []
	while True:
		optimizer.zero_grad()
		coords = a2c(x, length)
		rmsd = c2RMSD(coords, target_coords, length)
		rmsd_real = np.sqrt(rmsd.data[0])
		trace.append(rmsd_real)
		print rmsd_real
		if rmsd_real < 2.0:
			break
		rmsd.backward()
		optimizer.step()
	protein, target = rmsd.creator.get_aligned_coordinates()
	
	fig = plt.figure()
	plt.title("Fitting of the C-alpha model into protein coordinates")
	plt.plot(trace, 'r', label = 'rmsd trace')
	plt.xlabel("iteration")
	plt.ylabel("rmsd")
	plt.legend()
	plt.show()
	# plt.savefig('Fig/rmsd_trace.png')
	sx, sy, sz = get_structure(coords)
	protein = protein[0].numpy()
	target = target[0].numpy()
	# print protein
	sx, sy, sz = protein[:,0], protein[:,1], protein[:,2]
	rx, ry, rz = target[:,0], target[:,1], target[:,2]
	fig = plt.figure()
	plt.title("Fitted C-alpha model and the protein C-alpha coordinates")
	ax = p3.Axes3D(fig)
	ax.plot(sx,sy,sz, 'r', label = 'Fitted model')
	ax.plot(rx,ry,rz, '--b', label = 'Experimental structure')
	ax.legend()	
	plt.show()
	# plt.savefig('Fig/fitted_model.png')