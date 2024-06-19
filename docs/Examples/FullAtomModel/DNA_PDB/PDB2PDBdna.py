import Bio
import numpy
import torch
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
from torch import optim
import logging

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
# file = '/u2/home_u2/fam95/Documents/180d.pdb'
# file = '/u2/home_u2/fam95/Documents/6z0s.pdb'
file = '/u2/home_u2/fam95/Downloads/DNA/Batch1/1NZB.pdb'
# log_file = file[:-4] + file[27:32] + '.log'
# logging.basicConfig(filename=log_file, filemode='w', format='%(asctimes)s %(levelname)s:%(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
# logging.info('PDB: %s \n ', file[28:32])


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
loaded_na = p2c([file], chain_ids=['D'], polymer_types=polymer_type)
# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_na
coords_dst = coords_dst.to(dtype=torch.float)
saved_atom_names = atomnames

# logging.info('PDB: %s \n ', file[28:32]) log loaded pdb


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

sequences = get_sequence(resnames, resnums, num_atoms, mask, polymer_type)  # Needs Work??
print(sequences)

# Coords2Angles function Called
angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type)

# for i in range(len(resnames[0])):
#     print(_convert2str(resnames[0,i]).decode("utf-8"))

# Angles Saved as beforeAng for deltaAngle plot
before_ang = angles
# logging.info('PDB: %s \n ', file[28:32]) log before angles
#
# #
angles = angles.to(dtype=torch.float)
angles.requires_grad_()
optimizer = optim.Adam([angles], lr=0.001)
a2c = FullAtomModel.Angles2Coords()  # Current Work
pred_na = a2c(angles, sequences, chainnames, polymer_type)
coords_2, chainnames, resnames, resnums, atomnames, num_atoms = pred_na
# logging.info('PDB: %s \n ', file[28:32]) log optimizer(Adam) and learning rate

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

# ## A2C TEST Prints before and after coords (line 128 -> 142 if this is on 127)
# i  = 0
# while i < (len(coords_2[0])):
# #     print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
#     print(_convert2str(resnames[0, int(i/3)]).decode("utf-8"), _convert2str(atomnames[0, int(i/3)]).decode("utf-8"),
#           'x:', torch.Tensor.detach(coords_2[0, i]), ", y:", torch.Tensor.detach(coords_2[0, i+1]), ", z:", torch.Tensor.detach(coords_2[0, i+2]))
#     i += 3
# print(len(torch.Tensor.detach(coords_2)[0]))
#
# i  = 0
# while i < (len(coords_dst[0]) - 3):
# #     print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
#     print(_convert2str(saved_atom_names[0, int(i/3)]).decode("utf-8"), 'dst_x:', torch.Tensor.detach(coords_dst[0, i]),
#           ", y:", torch.Tensor.detach(coords_dst[0, i+1]), ", z:", torch.Tensor.detach(coords_dst[0, i+2]))
#     i += 3
# print(len(torch.Tensor.detach(coords_dst)[0]))

rmsd = RMSD.Coords2RMSD()

fig, ax = plt.subplots()
epochs = []
loss = []

#
for epoch in range(1):  # At 10 for test
    # print(epoch)
    epochs.append(epoch + 1)
    optimizer.zero_grad()
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames, polymer_type)
    L = rmsd(coords_src, coords_dst, num_atoms)
    # L.backward(polymer_type=polymer_type, chain_names=chainnames)
    L.backward()
    optimizer.step()
    loss_per = float(L)
    loss.append(loss_per)
    # print(loss_per)

coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames, polymer_type)
# i  = 0
# while i < (len(coords_src[0])):
#     # print(_convert2str(torch.Tensor.detach(coords_2[0, i])).decode("utf-8"))
#     print(_convert2str(resnames[0, int(i/3)]).decode("utf-8"), _convert2str(atomnames[0, int(i/3)]).decode("utf-8"),
#           'x:', torch.Tensor.detach(coords_src[0, i]), ", y:", torch.Tensor.detach(coords_src[0, i+1]), ", z:", torch.Tensor.detach(coords_src[0, i+2]))
#     i += 3
coords_aft = torch.Tensor.detach(coords_src)
after_ang, lengths = Coords2Angles(coords_aft, chainnames, resnames, resnums, atomnames, num_atoms,
                                   polymer_type)  # Does this step introduce new error?
# logging.info('PDB: %s \n ', file[28:32]) log after angles, loss, and epochs
# #
# new_coords, new_chainnames, new_resnames, new_resnums, new_atomnames, new_num_atoms = a2c(angles, sequences)
#
# # Name of new PDB file to be written
# pdb2pdb_na_test = '/u2/home_u2/fam95/Documents/pdb2pdb_na_test_119d.pdb'
pdb2pdb_dna_test = "/u2/home_u2/fam95/Documents/Test1NZB_chainD_lr001_RMSD0e0.pdb"  ## for 180d (next still need to do Pxy)
#
ax.plot(epochs, loss)
ax.set_ylim([0, 4])
ax.set_xlabel("epochs", fontsize=12)
ax.set_ylabel("rmsd (A)", fontsize=12)

# plt.savefig('/u2/home_u2/fam95/Documents/NAloss119d_0e0.png')  # [turned off for test] _ylim

# Creating DeltaAngle Plot
bef_det = torch.Tensor.detach(before_ang)
aft_det = torch.Tensor.detach(after_ang)
#
bef_array = np.array(bef_det)
aft_array = np.array(aft_det)
delta_angles = np.subtract(aft_array, bef_array)
#
# ##print(np.subtract(aftArray, befArray))
# alpha, beta, gamma, delta, epsilon, zeta, nu0, nu1, nu2, nu3, nu4, chi
delta_alp = delta_angles[0, 0]
delta_bet = delta_angles[0, 1]
delta_gam = delta_angles[0, 2]
delta_del = delta_angles[0, 3]
delta_eps = delta_angles[0, 4]
delta_zet = delta_angles[0, 5]
delta_nu0 = delta_angles[0, 6]
delta_nu1 = delta_angles[0, 7]
delta_nu2 = delta_angles[0, 8]
delta_nu3 = delta_angles[0, 9]
delta_nu4 = delta_angles[0, 10]
delta_chi = delta_angles[0, 11]
n_d_alp = []
n_d_bet = []
n_d_gam = []
n_d_del = []
n_d_eps = []
n_d_zet = []
n_d_nu0 = []
n_d_nu1 = []
n_d_nu2 = []
n_d_nu3 = []
n_d_nu4 = []
n_d_chi = []

for i in range(len(delta_alp)):
    d_alp_pl = delta_alp[i] + (2 * np.pi)
    d_alp_mi = delta_alp[i] - (2 * np.pi)
    d_bet_pl = delta_bet[i] + (2 * np.pi)
    d_bet_mi = delta_bet[i] - (2 * np.pi)
    d_gam_pl = delta_gam[i] + (2 * np.pi)
    d_gam_mi = delta_gam[i] - (2 * np.pi)
    d_del_pl = delta_del[i] + (2 * np.pi)
    d_del_mi = delta_del[i] - (2 * np.pi)
    d_eps_pl = delta_eps[i] + (2 * np.pi)
    d_eps_mi = delta_eps[i] - (2 * np.pi)
    d_zet_pl = delta_zet[i] + (2 * np.pi)
    d_zet_mi = delta_zet[i] - (2 * np.pi)
    d_nu0_pl = delta_nu0[i] + (2 * np.pi)
    d_nu0_mi = delta_nu0[i] - (2 * np.pi)
    d_nu1_pl = delta_nu1[i] + (2 * np.pi)
    d_nu1_mi = delta_nu1[i] - (2 * np.pi)
    d_nu2_pl = delta_nu2[i] + (2 * np.pi)
    d_nu2_mi = delta_nu2[i] - (2 * np.pi)
    d_nu3_pl = delta_nu3[i] + (2 * np.pi)
    d_nu3_mi = delta_nu3[i] - (2 * np.pi)
    d_nu4_pl = delta_nu4[i] + (2 * np.pi)
    d_nu4_mi = delta_nu4[i] - (2 * np.pi)
    d_chi_pl = delta_chi[i] + (2 * np.pi)
    d_chi_mi = delta_chi[i] - (2 * np.pi)

    n_d_alp_i = min(abs(delta_alp[i]), abs(d_alp_pl), abs(d_alp_mi))
    n_d_bet_i = min(abs(delta_bet[i]), abs(d_bet_pl), abs(d_bet_mi))
    n_d_gam_i = min(abs(delta_gam[i]), abs(d_gam_pl), abs(d_gam_mi))
    n_d_del_i = min(abs(delta_del[i]), abs(d_del_pl), abs(d_del_mi))
    n_d_eps_i = min(abs(delta_eps[i]), abs(d_eps_pl), abs(d_eps_mi))
    n_d_zet_i = min(abs(delta_zet[i]), abs(d_zet_pl), abs(d_zet_mi))
    n_d_nu0_i = min(abs(delta_nu0[i]), abs(d_nu0_pl), abs(d_nu0_mi))
    n_d_nu1_i = min(abs(delta_nu1[i]), abs(d_nu1_pl), abs(d_nu1_mi))
    n_d_nu2_i = min(abs(delta_nu2[i]), abs(d_nu2_pl), abs(d_nu2_mi))
    n_d_nu3_i = min(abs(delta_nu3[i]), abs(d_nu3_pl), abs(d_nu3_mi))
    n_d_nu4_i = min(abs(delta_nu4[i]), abs(d_nu4_pl), abs(d_nu4_mi))
    n_d_chi_i = min(abs(delta_chi[i]), abs(d_chi_pl), abs(d_chi_mi))

    # n_d_alp_i = n_d_alp_i % (2*np.pi)
    n_d_bet_i = n_d_bet_i % (2 * np.pi)
    # n_d_gam_i =
    # n_d_del_i =
    # n_d_eps_i =
    # n_d_zet_i =
    # n_d_nu0_i =
    # n_d_nu1_i =
    # n_d_nu2_i =
    # n_d_nu3_i =
    # n_d_nu4_i =
    # n_d_chi_i =
    if i == 12:  # or i ==0:
        n_d_alp.append(0)
    elif i != 12:  # or i != 0:
        n_d_alp.append(n_d_alp_i)

    if i == 12 or i == 0:
        n_d_bet.append(0)
    elif i != 12 or i != 0:
        n_d_bet.append(n_d_bet_i)

    n_d_gam.append(n_d_gam_i)

    n_d_del.append(n_d_del_i)

    if i == 11 or i == 23:
        n_d_eps.append(0)
    elif i != 11 or i != 23:
        n_d_eps.append(n_d_eps_i)

    n_d_zet.append(n_d_zet_i)

    n_d_nu0.append(n_d_nu0_i)

    n_d_nu1.append(n_d_nu1_i)

    n_d_nu2.append(n_d_nu2_i)

    n_d_nu3.append(n_d_nu3_i)

    n_d_nu4.append(n_d_nu4_i)

    n_d_chi.append(n_d_chi_i)
#
x_labels = []

for i in range(len(sequences[0])):
    sequence = sequences[0]
    # x_labels.append(sequence[i] + i)
    x_labels.append(i)
#
# # print(delta_phi)
# # print(delta_psi)
# # print(delta_omega)
# # print(y_angles)
#
# # Print length of the X array and the Y values for the Delta Angle Plot [TEST]
# print(len(x_labels))
# print((x_labels))
# print(n_d_alp)
# print(n_d_bet)
# print(n_d_gam)
# print(n_d_del)
# print(n_d_eps)
# print(n_d_zet)
# print(n_d_nu0)
# print(n_d_nu1)
# print(n_d_nu2)
# print(n_d_nu3)
# print(n_d_nu4)
# print(n_d_chi)
#
fig, ax = plt.subplots()
#
ax.plot(x_labels, n_d_alp, "r.-")
ax.plot(x_labels, n_d_bet, "g.-")
ax.plot(x_labels, n_d_gam, "b.-")
ax.plot(x_labels, n_d_del, "c.-")
ax.plot(x_labels, n_d_eps, "m.-")
ax.plot(x_labels, n_d_zet, "y.-")
# ax.plot(x_labels, n_d_nu0, "r.-")
# ax.plot(x_labels, n_d_nu1, "g.-")
# ax.plot(x_labels, n_d_nu2, "b.-")
# ax.plot(x_labels, n_d_nu3, "c.-")
# # ax.plot(x_labels, n_d_nu4, "m.-")
# ax.plot(x_labels, n_d_chi, "y.-")

ax.set_xlabel("Residue", fontsize=12)
ax.set_ylabel("Change in Angle", fontsize=12)
# ax.set_ylim([0, 0.01])


# plt.savefig('/u2/home_u2/fam95/Documents/119dangles1_chi2_5e4.png')  # [turned off for test]
#
## Save New PDB [turned off for test]
# print(num_atoms)
# logging.info('PDB: %s \n ', file[28:32]) log after coords
FullAtomModel.writePDB(pdb2pdb_dna_test, coords_src, chainnames, resnames, resnums, atomnames, num_atoms)
# FullAtomModel.writePDB(pdb2pdb_dna_test, coords_2, chainnames, resnames, resnums, atomnames, num_atoms)
# FullAtomModel.writePDB(pdb2pdb_dna_test, new_coords, chainnames, resnames, resnums, atomnames, num_atoms)
