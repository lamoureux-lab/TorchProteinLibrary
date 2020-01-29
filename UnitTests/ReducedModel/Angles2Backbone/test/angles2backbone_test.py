import sys
import os

import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import Bio.PDB
from Bio.PDB import calc_angle, rotaxis, Vector, calc_dihedral
from math import *
import numpy as np

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3

from TorchProteinLibrary.ReducedModel import Angles2Backbone
from TorchProteinLibrary.RMSD import Coords2RMSD
from TorchProteinLibrary.FullAtomModel import Coords2Center, CoordsTranslate, CoordsRotate


def getBackbone(residues):
	phi = [0.0]
	psi = []
	omega = []
	coords = torch.zeros(1, 3*len(residues)*3, dtype=torch.double, device='cpu')
	for i, res_i in enumerate(residues):
		N_i = res_i["N"].get_vector()
		coords[0, 9*i:9*i + 3] = torch.tensor([N_i[0], N_i[1], N_i[2]])
		CA_i = res_i["CA"].get_vector()
		coords[0, 9*i + 3:9*i + 6] = torch.tensor([CA_i[0],CA_i[1],CA_i[2]])
		C_i = res_i["C"].get_vector()
		coords[0, 9*i + 6:9*i + 9] = torch.tensor([C_i[0],C_i[1],C_i[2]])
		if i>0:
			res_im1 = residues[i-1]
			C_im1 = res_im1["C"].get_vector()
			phi.append(calc_dihedral(C_im1, N_i, CA_i, C_i))
		if i<(len(residues)-1):
			res_ip1 = residues[i+1]
			N_ip1 = res_ip1["N"].get_vector()
			psi.append(calc_dihedral(N_i, CA_i, C_i, N_ip1))
		if i<(len(residues)-1):
			res_ip1 = residues[i+1]
			CA_ip1 = res_ip1["CA"].get_vector()
			omega.append(calc_dihedral(CA_i, C_i, N_ip1, CA_ip1))
	psi.append(0.0)
	omega.append(0.0)
	return phi, psi, omega, coords

def getAngles(structure):
	residues = list(structure.get_residues())
	angles = torch.zeros(1, 3, len(residues), dtype=torch.double, device='cpu')
	phi, psi, omega, coords = getBackbone(residues)
	for i, residue in enumerate(residues):
		angles[0, 0, i] = phi[i]
		angles[0, 1, i] = psi[i]
		angles[0, 2, i] = omega[i]
	return angles, coords

def plot_coords(coords, name, line='-', ax=None):
	
	L = int(coords.size(0)/(3*3))
	coords = coords.view(3*L, 3)
	coords = coords.numpy()
	sx, sy, sz = coords[:,0], coords[:,1], coords[:,2]
	if ax is None:
		fig = plt.figure()
		ax = p3.Axes3D(fig)
	ax.plot(sx, sy, sz, line, label = name)
	
	ax_min_x, ax_max_x = ax.get_xlim()
	ax_min_y, ax_max_y = ax.get_ylim()
	ax_min_z, ax_max_z = ax.get_zlim()

	if ax is None:
		#Preserving aspect ratio
		min_x = min(np.min(sx).item(), ax_min_x)
		max_x = max(np.max(sx).item(), ax_max_x)
		min_y = min(np.min(sy).item(), ax_min_y)
		max_y = max(np.max(sy).item(), ax_max_y)
		min_z = min(np.min(sz).item(), ax_min_z)
		max_z = max(np.max(sz).item(), ax_max_z)
		max_L = max([max_x - min_x, max_y - min_y, max_z - min_z])
		ax.set_xlim(min_x, min_x+max_L)
		ax.set_ylim(min_y, min_y+max_L)
		ax.set_zlim(min_z, min_z+max_L)

	ax.legend()
	return ax

def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens,dev="cpu"): ### function taken from https://github.com/conradry/pytorch-rgn

	for i in range(3):
		#coordinates of last three atoms
		A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]

		#internal coordinates
		T = bond_angles[i]
		R = bond_lens[i]
		P = pred_torsions[:, i]
		
		#6x3 one triplet for each sample in the batch
		D2 = torch.stack([-R*torch.ones(P.size()).to(dev)*torch.cos(T), 
						  R*torch.cos(P)*torch.sin(T),
						  R*torch.sin(P)*torch.sin(T)], dim=1)

		#bsx3 one triplet for each sample in the batch
		BC = C - B
		bc = BC/torch.norm(BC, 2, dim=1, keepdim=True)

		AB = B - A

		N = torch.cross(AB, bc)
		n = N/torch.norm(N, 2, dim=1, keepdim=True)

		M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

		D = torch.bmm(M, D2.view(-1,3,1)).squeeze() + C
		pred_coords = torch.cat([pred_coords, D.view(1,-1,3)])
	return pred_coords

def rgn_angles2coords(angles):
	dev="cpu"
	batch_sz=1
	avg_bond_lens = torch.tensor([1.329, 1.459, 1.525]).to(dev)
	avg_bond_angles = torch.tensor([2.034, 2.119, 1.937]).to(dev)
	A = torch.tensor([0., 0., 1.]).to(dev)
	B = torch.tensor([0., 1., 0.]).to(dev)
	C = torch.tensor([1., 0., 0.]).to(dev)
	broadcast = torch.ones((batch_sz, 3)).to(dev)
	pred_coords = torch.stack([A*broadcast, B*broadcast, C*broadcast])
	for ix, triplet in enumerate(angles[:]):
		pred_coords = geometric_unit(pred_coords, triplet, 
									 avg_bond_angles, 
									 avg_bond_lens,dev=dev)
	
	pred_coords=pred_coords.transpose(0,1).squeeze(0)#.data.numpy()
	return pred_coords

def align(coords_src, coords_dst, num_atoms):
	#Function for aligning two structures
	center = Coords2Center()
	translate = CoordsTranslate()
	rotate = CoordsRotate()
	rmsd = Coords2RMSD()

	#aligning coords we generated to the referece structure
	loss = rmsd(coords_src, coords_dst, num_atoms)

	#centering two structures
	center_src = center(coords_src, num_atoms)
	center_dst = center(coords_dst, num_atoms)
	c_src = translate(coords_src, -center_src, num_atoms)
		
	#rotating structure we generated
	rc_src = rotate(c_src, rmsd.UT.transpose(1,2).contiguous(), num_atoms)
	rc_t_src = translate(rc_src, center_dst, num_atoms)

	return rc_t_src, loss.item()

if __name__=='__main__':
	crop_length = 100
	#Reading angles of the protein
	parser = PDBParser()
	structure = parser.get_structure('test', 'small.pdb')
	angles, coords_biopython = getAngles(structure)
	angles = angles[:,:,:crop_length].contiguous()
	coords_biopython = coords_biopython[:,:3*crop_length*3].contiguous()
	
	#Recovering coordinates from angles we read in structure
	a2b = Angles2Backbone()
	# param = a2b.get_default_parameters()
	param = torch.tensor([1.4360, 1.0199, 1.5187, 1.1855, 1.3197, 1.0543], dtype=torch.double, device='cpu')
	angles = angles.to(dtype=torch.double, device='cpu')
	length = torch.tensor([angles.size(2)], dtype=torch.int, device='cpu')
	coords_a2b = a2b(angles, param, length)

	
	src, rmsd = align(coords_a2b, coords_biopython, 3*length)
	print('RMSD:', rmsd)
	ax = plot_coords(coords_biopython[0,:], name='Reference', line='-')
	plot_coords(src[0,:], name='a2b CPU', line='-.', ax=ax)
	# plt.show()

	angles=  [[2.0393507,-3.08193687,-1.63904282],
			[1.88758931,3.13580149,-1.71118982],
			[0.35379004,3.08612295,-0.87478756],
			[-0.8281743,-3.01557253,-1.71838046],
			[0.08055055,3.1023487,-2.13224391],
			[-0.26218354,-3.13150822,-1.54592689],
			[2.01053773,-3.06474198,-1.81015267],
			[2.28223663,2.96027269,-1.4906815,],
			[2.26022762,-3.07716086,-2.11658479],
			[1.7943145,3.12770495,-1.81855047],
			[2.95888264,3.12862793,1.17557892],
			[0.42450582,3.10929074,-1.23104228],
			[2.56065955,2.9451381,-1.15394783],
			[2.61357919,-3.03015535,-1.17941288],
			[-0.22816098,3.09113235,-1.57717264],
			[-0.03619108,3.00468132,-1.12194906],
			[2.37864287,3.11036379,-2.00828385],
			[2.970595,2.9962374,-0.88871536],
			[-0.80831791,-3.05090645,-1.18386207],
			[-0.70809824,-3.09181413,-1.35472934],
			[-0.57338133,2.97493819,-1.03364934]]
	angles = angles[:crop_length]
	angles=torch.tensor(angles).unsqueeze(0).transpose(1,0).contiguous()
	coords_rgn = rgn_angles2coords(angles).view(1,3*length.item()*3).to(dtype=torch.double, device='cpu')
	src, rmsd = align(coords_rgn, coords_biopython, 3*length)
	print('RMSD:', rmsd)
	plot_coords(src[0,:], name='rgn CPU', line='-', ax=ax)
	plt.show()
	plt.savefig('TPL_test.png')

	
