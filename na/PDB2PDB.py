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
file = '/u2/home_u2/fam95/Documents/1a0b.pdb'

# Functions used to Get Sequence (Not Currently Working)
def _convert2str(tensor):
	return tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]

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

# PDB2Coords Function Called to Load Struct from File
p2c = FullAtomModel.PDB2CoordsOrdered()
loaded_prot = p2c([file])

# Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_prot
coords_dst = coords_dst.to(dtype=torch.float)

#Sequence Entered Manually until get sequence is fixed
sequences2 = [
    'KSEALLDIPMLEQYLELVGPKLITDGLAVFEKMMPGYVSVLESNLTAQDKKGIVEEGHKIKGAAGSVGLRHLQQLGQQIQSPDLPAWEDNVGEWIEEMKEEWRHDVEVLKAWVAKAT']
sequences = get_sequence(resnames, resnums, num_atoms, mask)
print(sequences)
print(sequences2)

# Coords2Angles function Called
angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms)


# Angles Saved as beforeAng for deltaAngle plot
beforeAng = angles

#
angles = angles.to(dtype=torch.float)
angles.requires_grad_()
optimizer = optim.Adam([angles], lr=0.00001)
a2c = FullAtomModel.Angles2Coords()
pred_prot = a2c(angles, sequences)
# print(angles, lengths)
# print(coords_dst)
rmsd = RMSD.Coords2RMSD()
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

fig, ax = plt.subplots()
epochs = []
loss = []

#
for epoch in range(40000):
    epochs.append(epoch + 1)
    optimizer.zero_grad()
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences2)
    L = rmsd(coords_src, coords_dst, num_atoms)
    L.backward()
    optimizer.step()
    lossPer = float(L)
    loss.append(lossPer)
    # scheduler.step()


# Coords after optimization converted to angles to save as afterAng for deltaAngle plot
coordsAft = torch.Tensor.detach(coords_src)
afterAng, lengths = Coords2Angles(coordsAft, chainnames, resnames, resnums, atomnames,
                                  num_atoms)  # Does this step introduce new error?

new_coords, new_chainnames, new_resnames, new_resnums, new_atomnames, new_num_atoms = a2c(angles, sequences2)

# Name of new PDB file to be written
pdb2pdbtest = '/u2/home_u2/fam95/Documents/pdb2pdbtest.pdb'
pdb2wrmsdtest = '/u2/home_u2/fam95/Documents/pdb2pdbopt_e1_lr0,0001.pdb'


# Creating Epoch vs RMSD plot
# print(epochs, loss)


ax.plot(epochs, loss)
ax.set_ylim([0,1])
ax.set_xlabel("epochs", fontsize=12)
ax.set_ylabel("rmsd (A)", fontsize=12)

plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_lossplt_ylim1_test4e4.png')

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
    sequence = sequences2[0]
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

# plt.savefig('/u2/home_u2/fam95/Documents/pdb2pdb_angleplt_e&h_test4e3.png')

# Save New PDB
# FullAtomModel.writePDB(pdb2wrmsdtest, coords_src, new_chainnames, new_resnames, new_resnums, new_atomnames, new_num_atoms)
