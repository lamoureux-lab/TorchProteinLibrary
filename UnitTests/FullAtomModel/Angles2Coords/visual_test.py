"""
Simulation of a block connected 
to a spring in a vicous medium.
"""
import torch
from torch import optim
import numpy as np

import vtkplotter as vp

from TorchProteinLibrary.Utils import ProteinStructure
from TorchProteinLibrary.FullAtomModel import PDB2CoordsOrdered, PDB2CoordsUnordered, Angles2Coords, Coords2Center, CoordsRotate, CoordsTranslate
from TorchProteinLibrary.RMSD import Coords2RMSD

from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1

from TorchProteinLibrary.FullAtomModel import Coords2Angles

def _convert2str(tensor):
	return tensor.numpy().astype(dtype=np.uint8).tostring().split(b'\00')[0]

def get_sequence(res_names, res_nums, num_atoms, mask):
	batch_size = res_names.size(0)
	sequences = []
	for batch_idx in range(batch_size):
		sequence = ""
		previous_resnum = res_nums[batch_idx, 0].item()
		for atom_idx in range(num_atoms[batch_idx].item()):
			if mask[batch_idx, atom_idx].item() == 0: continue
			if previous_resnum < res_nums[batch_idx, atom_idx].item():
				residue_name = _convert2str(res_names[batch_idx, atom_idx, :]).decode("utf-8")
				sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
				previous_resnum = res_nums[batch_idx, atom_idx].item()
		
		residue_name = _convert2str(res_names[batch_idx, -1, :]).decode("utf-8")
		sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
		sequences.append(sequence)
	return sequences


if __name__=='__main__':
	a2c = Angles2Coords()
	rmsd = Coords2RMSD()
	
	center = Coords2Center()
	translate = CoordsTranslate()
	rotate = CoordsRotate()


	
	p2c = PDB2CoordsOrdered()
	loaded_prot = p2c(["f4TQ1_B.pdb"])

	coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_prot
	coords_dst = coords_dst.to(dtype=torch.float)
	sequences = get_sequence(resnames, resnums, num_atoms, mask)
	
	angles, length = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms)
	angles = angles.to(dtype=torch.float)
	angles.requires_grad_()
	optimizer = optim.Adam([angles], lr = 0.001)
	pred_prot = a2c(angles, sequences)
	

	v = vp.Plotter(N=2, title='basic shapes', axes=0)
	v.sharecam = True
	for epoch in range(300):
		coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences)
		L = rmsd(coords_src, coords_dst, num_atoms)
		L.backward()
		optimizer.step()

		ref_structure = ProteinStructure(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms)
		pred_structure = ProteinStructure(coords_src, chainnames, resnames, resnums, atomnames, num_atoms)
		ref_atoms_plot = ref_structure.plot_atoms()
		pred_atoms_plot = pred_structure.plot_atoms()
		v.show(ref_atoms_plot, at=0)
		v.show(pred_atoms_plot, at=1)
	v.show(interactive=1)
# L = 0.1  # spring x position at rest
# x0 = 0.85  # initial x-coordinate of the block
# k = 25  # spring constant
# m = 20  # block mass
# b = 0.5  # viscosity friction (proportional to velocity)
# dt = 0.15  # time step

# # initial conditions
# v = vector(0, 0, 0.2)
# x = vector(x0, 0, 0)
# xr = vector(L, 0, 0)
# sx0 = vector(-0.8, 0, 0)
# offx = vector(0, 0.3, 0)

# vp += Box(pos=(0, -0.1, 0), length=2.0, width=0.02, height=0.5)  # surface
# vp += Box(pos=(-0.82, 0.15, 0), length=0.04, width=0.50, height=0.3)  # wall

# block = Cube(pos=x, side=0.2, c="tomato")
# block.addTrail(offset=[0, 0.2, 0], alpha=0.6, lw=2, n=500)
# spring = Spring(sx0, x, r=0.06, thickness=0.01)
# vp += [block, spring, Text(__doc__)]

# pb = ProgressBar(0, 300, c="r")
# for i in pb.range():
# 	F = -k * (x - xr) - b * v  # Force and friction
# 	a = F / m  # acceleration
# 	v = v + a * dt  # velocity
# 	x = x + v * dt + 1 / 2 * a * dt ** 2  # position

# 	block.pos(x)  # update block position and trail
# 	spring.stretch(sx0, x)  # stretch helix accordingly

# 	vp.show(elevation=0.1, azimuth=0.1)
# 	pb.print()

# vp.show(interactive=1)

