import Bio
import numpy
import torch
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
from torch import optim

from TorchProteinLibrary import FullAtomModel
from TorchProteinLibrary import RMSD
# import PDBloader
from Bio.PDB import *
import numpy as np

from TorchProteinLibrary.FullAtomModel import Coords2Angles

import matplotlib.pylab as plt

# Load File from Document Drive
file = '/u2/home_u2/fam95/Documents/119d.pdb'

# Functions used to Get Sequence
def _convert2str(tensor):
    return tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]

def get_sequence(res_names, res_nums, num_atoms, mask, polymer_type):
    batch_size = res_names.size(0)
    sequences = []
    for batch_idx in range(batch_size):
        sequence = ""
        previous_resnum = res_nums[batch_idx, 0].item() - 1
        for atom_idx in range(num_atoms[batch_idx].item()):
            if mask[batch_idx, atom_idx].item() == 0: continue
            if previous_resnum < res_nums[batch_idx, atom_idx].item():
                residue_name = _convert2str(res_names[batch_idx, atom_idx, :]).decode("utf-8")
                if polymer_type == 0:
                    sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                    previous_resnum = res_nums[batch_idx, atom_idx].item()
                elif polymer_type == 1:
                    # Either split string so that A,T,C, or G is added to Seq or if res == "DA,.." then seq += "A,.."
                    one_from_three = residue_name[1]
                    sequence = sequence + one_from_three
                    previous_resnum = res_nums[batch_idx, atom_idx].item()
                    # print("get_sequence not implemented for polymer type 1")
                elif polymer_type == 2:
                    # sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                    # previous_resnum = res_nums[batch_idx, atom_idx].item()
                    print("get_sequence not implemented for polymer type 2")


        residue_name = _convert2str(res_names[batch_idx, -1, :]).decode("utf-8")
        if polymer_type == 0:
            sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
            sequences.append(sequence[:-1])
        elif polymer_type == 1:
            print(residue_name)
            # one_from_three = residue_name[1]
            # print(one_from_three)
            # sequence = sequence + one_from_three
            sequences.append(sequence)
        # elif polymer_type == 1:
        #     one_from_three = residue_name
        #     sequence = sequence + one_from_three

    return sequences

# PDB2Coords Function Called to Load Struct from File
polymerType = 1
p2c = FullAtomModel.PDB2CoordsOrdered()
loaded_prot = p2c([file], polymer_type=polymerType)

# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_prot
coords_dst = coords_dst.to(dtype=torch.float)

# for i in range(num_atoms):
#     print(_convert2str(atomnames[0,i]).decode("utf-8"))
# print(num_atoms)

sequences = get_sequence(resnames, resnums, num_atoms, mask, polymerType)
print(sequences)

# Coords2Angles function Called
angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymerType)


# Angles Saved as beforeAng for deltaAngle plot
# beforeAng = angles
#
# #
# angles = angles.to(dtype=torch.float)
# angles.requires_grad_()
# optimizer = optim.Adam([angles], lr=0.00001)
# a2c = FullAtomModel.Angles2Coords()