import Bio
import numpy
import torch
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
from torch import optim

from TorchProteinLibrary import FullAtomModel
from TorchProteinLibrary import RMSD
import PDBloader
from Bio.PDB import *
import numpy as np

from TorchProteinLibrary.FullAtomModel import Coords2Angles

import matplotlib.pylab as plt

# Load File from Document Drive
file = '/u2/home_u2/fam95/Documents/119d.pdb'

# Functions used to Get Sequence
def _convert2str(tensor):
    return tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]

def get_sequence(res_names, res_nums, num_atoms, mask):
    batch_size = res_names.size(0)
    sequences = []
    for batch_idx in range(batch_size):
        sequence = ""
        previous_resnum = res_nums[batch_idx, 0].item() - 1
        for atom_idx in range(num_atoms[batch_idx].item()):
            if mask[batch_idx, atom_idx].item() == 0: continue
            if previous_resnum < res_nums[batch_idx, atom_idx].item():
                residue_name = _convert2str(res_names[batch_idx, atom_idx, :]).decode("utf-8")
                sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                previous_resnum = res_nums[batch_idx, atom_idx].item()

        residue_name = _convert2str(res_names[batch_idx, -1, :]).decode("utf-8")
        sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
        sequences.append(sequence[:-1])
    return sequences

# PDB2Coords Function Called to Load Struct from File
p2c = FullAtomModel.PDB2CoordsOrdered()
loaded_prot = p2c([file], polymer_type=1)

# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_prot
coords_dst = coords_dst.to(dtype=torch.float)

# sequences = get_sequence(resnames, resnums, num_atoms, mask)
# print(sequences)