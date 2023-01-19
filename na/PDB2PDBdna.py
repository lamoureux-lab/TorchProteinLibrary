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
# file = '/u2/home_u2/fam95/Documents/1d29.pdb'

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
polymerType = 1
p2c = FullAtomModel.PDB2CoordsOrdered()
loaded_prot = p2c([file], polymer_type=polymerType)

# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_prot
coords_dst = coords_dst.to(dtype=torch.float)

chain_num = 0
chain_name = _convert2str(chainnames[0,0]).decode("utf-8")

for i in range(len(_convert2str(chainnames[0,:]).decode("utf-8"))):
    if _convert2str(chainnames[0,i]).decode("utf-8") == chain_name:
        chain_num += 1

num_atoms -= chain_num

# print(chainnames)
# for i in range(num_atoms):
#     print(_convert2str(atomnames[0,i]).decode("utf-8"))
    # print((atomnames[0, i]))
print("num_atoms", num_atoms)

sequences = get_sequence(resnames, resnums, num_atoms, mask, polymerType) #Needs Work??
# print(sequences)

# Coords2Angles function Called
angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymerType)


# Angles Saved as beforeAng for deltaAngle plot
beforeAng = angles
#
# #
angles = angles.to(dtype=torch.float)
angles.requires_grad_()
optimizer = optim.Adam([angles], lr=0.00001)
a2c = FullAtomModel.Angles2Coords(polymer_type = polymerType) #Current Work
pred_prot = a2c(angles, sequences, polymerType, num_atoms)

rmsd = RMSD.Coords2RMSD()

fig, ax = plt.subplots()
epochs = []
loss = []

#
for epoch in range(10): #At 10 fo test
    epochs.append(epoch + 1)
    optimizer.zero_grad()
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, polymerType, num_atoms)
    L = rmsd(coords_src, coords_dst, num_atoms)
    L.backward()
    optimizer.step()
    lossPer = float(L)
    loss.append(lossPer)

coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences)
coordsAft = torch.Tensor.detach(coords_src)
afterAng, lengths = Coords2Angles(coordsAft, chainnames, resnames, resnums, atomnames,
                                  num_atoms)  # Does this step introduce new error?

new_coords, new_chainnames, new_resnames, new_resnums, new_atomnames, new_num_atoms = a2c(angles, sequences)

# Name of new PDB file to be written
pdb2pdbNAtest = '/u2/home_u2/fam95/Documents/pdb2pdbnatest.pdb'

ax.plot(epochs, loss)
ax.set_ylim([0,1])
ax.set_xlabel("epochs", fontsize=12)
ax.set_ylabel("rmsd (A)", fontsize=12)

# plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_lossplt_ylim1_TPLna_ehbb_test1e6.png') [turned off for test]

#Creating DeltaAngle Plot
befDet = torch.Tensor.detach(beforeAng)
aftDet = torch.Tensor.detach(afterAng)

befArray = np.array(befDet)
aftArray = np.array(aftDet)
deltaAngles = np.subtract(aftArray, befArray)

##print(np.subtract(aftArray, befArray))
delta_phi = deltaAngles[0, 0]
delta_psi = deltaAngles[0, 1]
delta_omega = deltaAngles[0, 2]
n_d_phi = []
n_d_psi = []
n_d_ome = []

for i in range(len(delta_phi)):
    d_phi_pl = delta_phi[i] + (2 * np.pi)
    d_phi_mi = delta_phi[i] - (2 * np.pi)
    d_psi_pl = delta_psi[i] + (2 * np.pi)
    d_psi_mi = delta_psi[i] - (2 * np.pi)
    d_ome_pl = delta_omega[i] + (2 * np.pi)
    d_ome_mi = delta_omega[i] - (2 * np.pi)

    n_d_phi_i = min(abs(delta_phi[i]), abs(d_phi_pl), abs(d_phi_mi))
    n_d_psi_i = min(abs(delta_psi[i]), abs(d_psi_pl), abs(d_psi_mi))
    n_d_ome_i = min(abs(delta_omega[i]), abs(d_ome_pl), abs(d_ome_mi))

    n_d_phi.append(n_d_phi_i)

    n_d_psi.append(n_d_psi_i)

    n_d_ome.append(n_d_ome_i)

x_labels = []

for i in range(len(sequences[0])):
    sequence = sequences[0]
    # x_labels.append(sequence[i] + i)
    x_labels.append(i)

# print(delta_phi)
# print(delta_psi)
# print(delta_omega)
# print(y_angles)

# Print length of the X array and the Y values for the Delta Angle Plot [TEST]
# print(len(x_labels))
# print(n_d_phi)
# print(n_d_psi)
# print(n_d_ome)

fig, ax = plt.subplots()

ax.plot(x_labels, n_d_ome, "r.-")
ax.plot(x_labels[1:], n_d_phi[1:], "g.-")
ax.plot(x_labels[:-2], n_d_psi[:-2], "b.-")

ax.set_xlabel("Residue", fontsize=12)
ax.set_ylabel("Change in Angle", fontsize=12)

# plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_angleplt_e&hbb_PLna_test1e6.png') [turned off for test]

# Save New PDB [turned off for test]
# FullAtomModel.writePDB(pdb2pdbNAtest, coords_dst, chainnames, resnames, resnums, atomnames, num_atoms)