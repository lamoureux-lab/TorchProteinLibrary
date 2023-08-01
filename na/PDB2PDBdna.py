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
# file = '/u2/home_u2/fam95/Documents/119d.pdb'
# file = '/u2/home_u2/fam95/Documents/1d29.pdb'
file = '/u2/home_u2/fam95/Documents/180d.pdb'
# file = '/u2/home_u2/fam95/Documents/6z0s.pdb'


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
            # print(residue_name)
            # one_from_three = residue_name[1]
            # print(one_from_three)
            # sequence = sequence + one_from_three
            sequences.append(sequence)
        # elif polymer_type == 1:
        #     one_from_three = residue_name
        #     sequence = sequence + one_from_three

    return sequences

# PDB2Coords Function Called to Load Struct from File
polymer_type = 1
p2c = FullAtomModel.PDB2CoordsOrdered()
loaded_na = p2c([file], polymer_type=polymer_type)

# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_na
coords_dst = coords_dst.to(dtype=torch.float)
saved_atom_names = atomnames

chain_num = 0
chain_name = _convert2str(chainnames[0, 0]).decode("utf-8")

for i in range(len(_convert2str(chainnames[0, :]).decode("utf-8"))):
    if _convert2str(chainnames[0, i]).decode("utf-8") == chain_name:
        chain_num += 1

num_atoms -= chain_num
# num_atoms_2 = num_atoms.item()
# print(num_atoms_2)

# print(len(chainnames[0, :]))
# for i in range(num_atoms):
#     print(_convert2str(atomnames[0,i]).decode("utf-8"))
#     print(atomnames[0, i])
# print("num_atoms", num_atoms)

sequences = get_sequence(resnames, resnums, num_atoms, mask, polymer_type) #Needs Work??
# print(sequences)

# Coords2Angles function Called
angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type)

# for i in range(len(resnames[0])):
#     print(_convert2str(resnames[0,i]).decode("utf-8"))

# Angles Saved as beforeAng for deltaAngle plot
before_ang = angles
#
# #
angles = angles.to(dtype=torch.float)
angles.requires_grad_()
optimizer = optim.Adam([angles], lr=0.00001)
a2c = FullAtomModel.Angles2Coords(polymer_type=polymer_type)  # Current Work
pred_na = a2c(angles, sequences, polymer_type, num_atoms, chainnames)
coords_2, chainnames, resnames, resnums, atomnames, num_atoms = pred_na

# print(resnames[0], atomnames[0])
# for i in range(len(resnames[0])):
#     print(_convert2str(resnames[0,i]).decode("utf-8"), _convert2str(atomnames[0,i]).decode("utf-8"))
    # print(resnames[0, i], atomnames[0, i])
    # print(atomnames[0, i])
#
# print(len(torch.Tensor.detach(coords_2)[0]))
# for i in range(len(coords_2[0])):
# #     print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
#     print('x:',torch.Tensor.detach(coords_2[0, i]), ", y:", torch.Tensor.detach(coords_2[0, i+1]), ", z:", torch.Tensor.detach(coords_2[0, i+2]))
#     i += 2

i  = 0
while i < (len(coords_2[0])):
#     print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
    print(_convert2str(resnames[0, int(i/3)]).decode("utf-8"), _convert2str(atomnames[0, int(i/3)]).decode("utf-8"),
          'x:', torch.Tensor.detach(coords_2[0, i]), ", y:", torch.Tensor.detach(coords_2[0, i+1]), ", z:", torch.Tensor.detach(coords_2[0, i+2]))
    i += 3
print(len(torch.Tensor.detach(coords_2)[0]))

i  = 0
while i < (len(coords_dst[0]) - 3):
#     print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
    print(_convert2str(saved_atom_names[0, int(i/3)]).decode("utf-8"), 'dst_x:', torch.Tensor.detach(coords_dst[0, i]),
          ", y:", torch.Tensor.detach(coords_dst[0, i+1]), ", z:", torch.Tensor.detach(coords_dst[0, i+2]))
    i += 3
print(len(torch.Tensor.detach(coords_dst)[0]))
# rmsd = RMSD.Coords2RMSD()
#
# fig, ax = plt.subplots()
# epochs = []
# loss = []
#
# #
# for epoch in range(10):  # At 10 for test
#     epochs.append(epoch + 1)
#     optimizer.zero_grad()
#     coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, polymer_type, num_atoms, chainnames)
#     L = rmsd(coords_src, coords_dst, num_atoms)
#     # L.backward(polymer_type=polymer_type, chain_names=chainnames)
#     L.backward()
#     optimizer.step()
#     loss_per = float(L)
#     loss.append(loss_per)
#
# # coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences)
# # coords_aft = torch.Tensor.detach(coords_src)
# # after_ang, lengths = Coords2Angles(coords_aft, chainnames, resnames, resnums, atomnames,
# #                                    num_atoms)  # Does this step introduce new error?
# #
# new_coords, new_chainnames, new_resnames, new_resnums, new_atomnames, new_num_atoms = a2c(angles, sequences)
#
# # Name of new PDB file to be written
# pdb2pdb_na_test = '/u2/home_u2/fam95/Documents/pdb2pdb_na_test_119d.pdb'
pdb2pdb_dna_test = "/u2/home_u2/fam95/Documents/pdb2pdb_na_test_180d_modbbring&chiang3.pdb" ## for 180d
#
# ax.plot(epochs, loss)
# # ax.set_ylim([0, 1])
# ax.set_xlabel("epochs", fontsize=12)
# ax.set_ylabel("rmsd (A)", fontsize=12)
#
# plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_lossplt__TPLdna_180d_test1e1.png') #[turned off for test]  ylim1
#
# #Creating DeltaAngle Plot
# bef_det = torch.Tensor.detach(before_ang)
# aft_det = torch.Tensor.detach(after_ang)
#
# bef_array = np.array(bef_det)
# aft_array = np.array(aft_det)
# delta_angles = np.subtract(aft_array, bef_array)
#
# ##print(np.subtract(aftArray, befArray))
# delta_phi = delta_angles[0, 0]
# delta_psi = delta_angles[0, 1]
# delta_omega = delta_angles[0, 2]
# n_d_phi = []
# n_d_psi = []
# n_d_ome = []
#
# for i in range(len(delta_phi)):
#     d_phi_pl = delta_phi[i] + (2 * np.pi)
#     d_phi_mi = delta_phi[i] - (2 * np.pi)
#     d_psi_pl = delta_psi[i] + (2 * np.pi)
#     d_psi_mi = delta_psi[i] - (2 * np.pi)
#     d_ome_pl = delta_omega[i] + (2 * np.pi)
#     d_ome_mi = delta_omega[i] - (2 * np.pi)
#
#     n_d_phi_i = min(abs(delta_phi[i]), abs(d_phi_pl), abs(d_phi_mi))
#     n_d_psi_i = min(abs(delta_psi[i]), abs(d_psi_pl), abs(d_psi_mi))
#     n_d_ome_i = min(abs(delta_omega[i]), abs(d_ome_pl), abs(d_ome_mi))
#
#     n_d_phi.append(n_d_phi_i)
#
#     n_d_psi.append(n_d_psi_i)
#
#     n_d_ome.append(n_d_ome_i)
#
# x_labels = []
#
# for i in range(len(sequences[0])):
#     sequence = sequences[0]
#     # x_labels.append(sequence[i] + i)
#     x_labels.append(i)
#
# # print(delta_phi)
# # print(delta_psi)
# # print(delta_omega)
# # print(y_angles)
#
# # Print length of the X array and the Y values for the Delta Angle Plot [TEST]
# # print(len(x_labels))
# # print(n_d_phi)
# # print(n_d_psi)
# # print(n_d_ome)
#
# fig, ax = plt.subplots()
#
# ax.plot(x_labels, n_d_ome, "r.-")
# ax.plot(x_labels[1:], n_d_phi[1:], "g.-")
# ax.plot(x_labels[:-2], n_d_psi[:-2], "b.-")
#
# ax.set_xlabel("Residue", fontsize=12)
# ax.set_ylabel("Change in Angle", fontsize=12)
#
# # plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_angleplt_e&hbb_PLna_test1e6.png') [turned off for test]
#
## Save New PDB [turned off for test]
# print(num_atoms)
# FullAtomModel.writePDB(pdb2pdb_dna_test, coords_2, chainnames, resnames, resnums, atomnames, num_atoms)
# FullAtomModel.writePDB(pdb2pdb_dna_test, new_coords, chainnames, resnames, resnums, atomnames, num_atoms)
