import torch
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1
from torch import optim
import logging

from TorchProteinLibrary import FullAtomModel
from TorchProteinLibrary import RMSD
import numpy as np
import pandas as pd
from TorchProteinLibrary.FullAtomModel import Coords2Angles
import matplotlib.pylab as plt
from pathlib import Path
import time

startTime = time.time()


# read csv and load into df



def parse_csv_to_df(filename):
    csv_df = pd.read_csv(filename, header=None, low_memory=False)
    print('csv_df \n', csv_df)
    return csv_df

# for each row of df, get pdb id, chain id, polymer type
def get_pdbinfo_from_df(csv_df):
    pdb_ids = []
    chain_ids = []
    poly_types = []
    csv_array = csv_df.to_numpy()
    print(csv_array)
    for row in csv_array:
        print(row)
        pdb_id = row[2]
        chain_id = row[3]
        polymer_type = row[4]
        if polymer_type == 'DNA':
            poly_type = 1
        if polymer_type == 'RNA':
            poly_type = 2
        pdb_ids.append(pdb_id)
        chain_ids.append(chain_id)
        poly_types.append(poly_type)
        print('pdb_id', pdb_id, 'chain_id', chain_id, 'poly_type', poly_type)

    return pdb_ids, chain_ids, poly_types

# log1 pdbids, chainids, and polymertypes


# put pdb ids into file path and then load pdb files using pdb2coords
def get_file_path_from_pdb_ids(pdb_ids):
    file_paths = []
    if type(pdb_ids) == str:
        file = '/u2/home_u2/fam95/Downloads/RNA/Batch4/' + pdb_ids + '.pdb'
        # print('file', file)
        file_paths.append(file)

    if type(pdb_ids) == list:
        for pdb_id in pdb_ids:
            file = '/u2/home_u2/fam95/Downloads/RNA/Batch4/'+ pdb_id + '.pdb'
            # print('file', file)
            file_paths.append(file)

    return file_paths

# get only chains associated with chain ids
def parse_pdbs(file_paths, chain_ids, poly_types):
    p2c = FullAtomModel.PDB2CoordsOrdered()
    loaded_na = p2c(file_paths, chain_ids, poly_types) #need to add arg to PDB2Coords that will limit parsing of pdb to desired chain

    # Coords, Chains, Residue names and numbers, Atoms, and total number of atoms loaded from structure
    coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms = loaded_na
    coords_dst = coords_dst.to(dtype=torch.float)

    return coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms, poly_types


# get sequences
def _convert2str(tensor):
    return tensor.numpy().astype(dtype=np.uint8).tobytes().split(b'\00')[0]


def get_sequence(res_names, res_nums, num_atoms, mask, polymer_types):
    batch_size = res_names.size(0)
    sequences = []
    for batch_idx in range(batch_size):
        sequence = ""
        previous_resnum = res_nums[batch_idx, 0].item() - 1
        for atom_idx in range(num_atoms[batch_idx].item()):
            if mask[batch_idx, atom_idx].item() == 0: continue
            if previous_resnum < res_nums[batch_idx, atom_idx].item():
                residue_name = _convert2str(res_names[batch_idx, atom_idx, :]).decode("utf-8")
                if type(polymer_types) == int:
                    if polymer_types == 0:
                        sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                        previous_resnum = res_nums[batch_idx, atom_idx].item()
                    elif polymer_types == 1:
                        # Either split string so that A,T,C, or G is added to Seq or if res == "DA,.." then seq += "A,.."
                        one_from_three = residue_name[1]
                        sequence = sequence + one_from_three
                        previous_resnum = res_nums[batch_idx, atom_idx].item()
                        # print("get_sequence not implemented for polymer type 1")
                    elif polymer_types == 2:
                        one_from_three = residue_name[0]
                        sequence = sequence + one_from_three
                        previous_resnum = res_nums[batch_idx, atom_idx].item()
                if type(polymer_types) == list:
                    if polymer_types[batch_idx] == 0:
                        sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                        previous_resnum = res_nums[batch_idx, atom_idx].item()
                    elif polymer_types[batch_idx] == 1:
                        # Either split string so that A,T,C, or G is added to Seq or if res == "DA,.." then seq += "A,.."
                        one_from_three = residue_name[1]
                        sequence = sequence + one_from_three
                        previous_resnum = res_nums[batch_idx, atom_idx].item()
                        # print("get_sequence not implemented for polymer type 1")
                    elif polymer_types[batch_idx] == 2:
                        one_from_three = residue_name[0]
                        sequence = sequence + one_from_three
                        previous_resnum = res_nums[batch_idx, atom_idx].item()

        residue_name = _convert2str(res_names[batch_idx, -1, :]).decode("utf-8")
        if type(polymer_types) == int:
            if polymer_types == 0:
                sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                sequences.append(sequence[:-1])
            elif polymer_types == 1 or polymer_types == 2:
                sequences.append(sequence)
        if type(polymer_types) == list:
            if polymer_types[batch_idx] == 0:
                sequence = sequence + dindex_to_1[d3_to_index[residue_name]]
                sequences.append(sequence[:-1])
            elif polymer_types[batch_idx] == 1 or polymer_types[batch_idx] == 2:
                # print(residue_name)
                # one_from_three = residue_name[1]
                # print(one_from_three)
                # sequence = sequence + one_from_three
                sequences.append(sequence)
            # elif polymer_type == 1:
            #     one_from_three = residue_name
            #     sequence = sequence + one_from_three

    return sequences

# angles from coords2angles
def get_angles_from_coords(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymer_types):
    angles, lengths = Coords2Angles(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, polymer_types)
    return angles, lengths

# log2 before angles and store as variable for plotting

# set up optimizer for angles and then get new coords from angles2coords
def optimize_angles(angles,sequences,chainnames, polymer_types, coords_dst, pdb_ids):
    angles = angles.to(dtype=torch.float)
    angles.requires_grad_()
    optimizer = optim.Adam([angles], lr=0.001)
    a2c = FullAtomModel.Angles2Coords()
    pred_na = a2c(angles, sequences, chainnames, polymer_types)
    coords_2, chainnames, resnames, resnums, atomnames, num_atoms = pred_na


# log3 optimizer(Adam) and learning rate

    # set up loss per epoch plot and RMSD function then call rmsd optimizer over x epochs
    rmsd = RMSD.Coords2RMSD()

    # fig, ax = plt.subplots()
    epochs = []
    loss = []
    loss_at1000 = {}
    loss_at2000 = {}
    loss_at4000 = {}

    for epoch in range(1000):  # repeat to at least 4000
        # print(epoch)
        epochs.append(epoch + 1)
        optimizer.zero_grad()
        coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames,
                                                                              polymer_types)
        L = rmsd(coords_src, coords_dst, num_atoms)
        # L.backward(polymer_type=polymer_type, chain_names=chainnames)
        L.backward()
        optimizer.step()
        loss_per = float(L)
        if epoch == 999:
            loss_at1000[pdb_ids] = [pdb_ids, loss_per]
        loss.append(loss_per)
        # print(loss_per)

    # save pdb and loss per x epochs then repeat as many times as desired
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames, polymer_types)
    coords_aft1000 = torch.Tensor.detach(coords_src)
    write_pdb(pdb_ids, coords_aft1000, chainnames, resnames, resnums, atomnames, num_atoms, 1000, loss_at1000[pdb_ids][1])

    for epoch in range(1000):  # repeat to at least 4000
        # print(epoch)
        epochs.append(epoch + 1001)
        optimizer.zero_grad()
        coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames,
                                                                              polymer_types)
        L = rmsd(coords_src, coords_dst, num_atoms)
        # L.backward(polymer_type=polymer_type, chain_names=chainnames)
        L.backward()
        optimizer.step()
        loss_per = float(L)
        if epoch == 999:
            for i in range(len(L)):
                loss_at2000[pdb_ids] = [pdb_ids, loss_per]
        loss.append(loss_per)
        # print(loss_per)

    # save pdb and loss per x epochs then repeat as many times as desired
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames, polymer_types)
    coords_aft2000 = torch.Tensor.detach(coords_src)
    write_pdb(pdb_ids, coords_aft2000, chainnames, resnames, resnums, atomnames, num_atoms, 2000, loss_at2000[pdb_ids][1])

    for epoch in range(2000):  # repeat to at least 4000
        # print(epoch)
        epochs.append(epoch + 2001)
        optimizer.zero_grad()
        coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames,
                                                                              polymer_types)
        L = rmsd(coords_src, coords_dst, num_atoms)
        # L.backward(polymer_type=polymer_type, chain_names=chainnames)
        L.backward()
        optimizer.step()
        loss_per = float(L)
        if epoch == 1999:
            for i in range(len(L)):
                loss_at4000[pdb_ids] = [pdb_ids, loss_per]
        loss.append(loss_per)
        # print(loss_per)

    # save pdb and loss per x epochs then repeat as many times as desired
    coords_src, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences, chainnames, polymer_types)
    coords_aft4000 = torch.Tensor.detach(coords_src)
    write_pdb(pdb_ids, coords_aft4000, chainnames, resnames, resnums, atomnames, num_atoms, 4000, loss_at4000[pdb_ids][1])
    # save pdb and loss per x epochs then repeat as many times as desired
    after_ang, lengths = Coords2Angles(coords_aft4000, chainnames, resnames, resnums, atomnames, num_atoms, polymer_types)
    return coords_aft4000, chainnames, resnames, resnums, atomnames, num_atoms, epochs, loss, after_ang, lengths, loss_at1000, loss_at2000, loss_at4000

# log4 after angles, loss, and epochs

# save loss vs epoch plot
def loss_per_epoch_plot(loss,epochs, pdb_ids):
    fig, ax = plt.subplots()
    ax.plot(epochs, loss)
    # ax.set_ylim([0, 4])
    ax.set_xlabel("epochs", fontsize=12)
    ax.set_ylabel("rmsd (A)", fontsize=12)

    plt.savefig('/u2/home_u2/fam95/Downloads/RNA/Batch4/' + pdb_ids + '/' + pdb_ids + 'lossplot_4e3.png')

# set up final path to save pdb and then write new pdb
def write_pdbs(pdb_ids, coords_src, chainnames, resnames, resnums, atomnames, num_atoms):
    new_file = '/u2/home_u2/fam95/Downloads/RNA/Batch4/' + pdb_ids + '/' + pdb_ids + ".pdb"
    FullAtomModel.writePDB(new_file, coords_src, chainnames, resnames, resnums, atomnames, num_atoms)

def write_pdb(pdb_id, coords_src, chainnames, resnames, resnums, atomnames, num_atoms, epoch, loss):
    print('loss', loss)
    new_file = Path('/u2/home_u2/fam95/Downloads/RNA/Batch4/' + pdb_id + '/' + pdb_id + '_' + str(epoch) + '_' + str(loss) + ".pdb")
    new_file.parent.mkdir(parents=True, exist_ok=True)
    FullAtomModel.writePDB(new_file, coords_src, chainnames, resnames, resnums, atomnames, num_atoms)

def write_pdbs_1(pdb_ids, coords_src, chainnames, resnames, resnums, atomnames, num_atoms,epoch, loss_list):
    batch_size = len(pdb_ids)
    for batch_idx in range(batch_size):
        new_file = Path('/u2/home_u2/fam95/Downloads/RNA/Batch4/' + pdb_ids[batch_idx] + '/' + pdb_ids[batch_idx] + str(epoch) + '_' + str(loss_list[pdb_ids[batch_idx]]) +".pdb")
        # print(new_file)
        new_file.parent.mkdir(parents=True, exist_ok=True)
        # print('coords:', coords_src, '\n chains:', chainnames, '\n resnames', resnames, '\n resnums', resnums,
        #       '\n atomnames:', atomnames, '\n numatoms:', num_atoms[batch_idx])
        # print('coords[idx]:', coords_src[batch_idx], '\n chains[idx]:', chainnames[batch_idx], '\n resnames[idx]', resnames[batch_idx],
        #       '\n resnums[idx]', resnums[batch_idx], '\n atomnames[idx]:', atomnames[batch_idx], '\n numatoms[idx]:', num_atoms[batch_idx])
        FullAtomModel.writePDB(new_file, coords_src[batch_idx], chainnames[batch_idx], resnames[batch_idx],
                               resnums[batch_idx], atomnames[batch_idx], num_atoms[batch_idx], add_model=False)

if __name__ == "__main__":
    #for each run change input[247] and output csv[303] location, change pdb batch locations[52],
    # check polymer type[52,225,230,235,240,247,259,303], and new pdb locations[230,235,240]
    #read csv and load into df
    csv = '/u2/home_u2/fam95/Downloads/RNA/Batch4/na_pdb_dataset_long_chains.csv'
    pdb_df = parse_csv_to_df(csv)

    #for each row of df, get pdb id, chain id, polymer type
    pdb_ids, chain_ids, poly_types = get_pdbinfo_from_df(pdb_df)

    batch_loss_at_1000 = {}
    batch_loss_at_2000 = {}
    batch_loss_at_4000 = {}

    # if not len(pdb_ids):
    #     # log1 pdbids, chainids, and polymertypes
    #
    #     # put pdb ids into file path and then load pdb files using pdb2coords
    #     file_paths = get_file_path_from_pdb_ids(pdb_ids)
    #
    #     # get only chains associated with chain ids
    #     polymer_type = 2
    #     coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms, poly_types = parse_pdbs(file_paths,
    #                                                                                                    chain_ids,
    #                                                                                                    polymer_type)
    #
    #     # get sequences
    #     sequences = get_sequence(resnames, resnums, num_atoms, mask, poly_types)
    #     print(sequences)
    #
    #     # angles from coords2angles
    #     angles, lengths = get_angles_from_coords(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms,
    #                                              poly_types)
    #     # log2 before angles and store as variable for plotting
    #
    #     # set up optimizer for angles and then get new coords from angles2coords
    #     coords_2, chainnames, resnames, resnums, atomnames, num_atoms, epochs, loss, after_ang, lengths, loss_at1000, \
    #     loss_at2000, loss_at4000 = optimize_angles(angles, sequences, chainnames, poly_types, coords_dst, pdb_ids)
    #
    #     # log3 optimizer(Adam) and learning rate
    #     print("loss_at1000", loss_at1000)
    #     # set up loss per epoch plot and RMSD function then call rmsd optimizer over x epochs
    #
    #     # save pdb and loss per x epochs then repeat as many times as desired
    #
    #     # log4 after angles, loss, and epochs
    #
    #     # save loss vs epoch plot
    #     loss_per_epoch_plot(loss, epochs, pdb_ids)
    #
    #     # set up final path to save pdb and then write new pdb
    #     write_pdbs(pdb_ids, coords_2, chainnames, resnames, resnums, atomnames, num_atoms)
    print(len(pdb_ids), type(pdb_ids), pdb_ids)
    for i in range(len(pdb_ids)):
        #log1 pdbids, chainids, and polymertypes
        #temp if statement to remove test dataset by removing pdbs that cause errors
        if pdb_ids[i] == '6WAZ' or pdb_ids[i] == '6YHS' or pdb_ids[i] == '6YT9' or pdb_ids[i] == '7F0D':
            print("continuing past" + pdb_ids[i])
            continue

        #put pdb ids into file path and then load pdb files using pdb2coords
        file_paths = get_file_path_from_pdb_ids(pdb_ids[i])
        print(chain_ids[i], file_paths)

        #get only chains associated with chain ids
        polymer_type = 2
        coords_dst, chainnames, resnames, resnums, atomnames, mask, num_atoms, poly_types = parse_pdbs(file_paths, chain_ids[i], polymer_type)
        # print('Z')
        #get sequences
        sequences = get_sequence(resnames, resnums, num_atoms, mask, poly_types)
        print(sequences)

        #angles from coords2angles
        angles, lengths = get_angles_from_coords(coords_dst, chainnames, resnames, resnums, atomnames, num_atoms, poly_types)
        #log2 before angles and store as variable for plotting

        #set up optimizer for angles and then get new coords from angles2coords
        coords_2, chainnames, resnames, resnums, atomnames, num_atoms, epochs, loss, after_ang, lengths, loss_at1000, \
        loss_at2000, loss_at4000 = optimize_angles(angles, sequences, chainnames, poly_types, coords_dst, pdb_ids[i])

        #log3 optimizer(Adam) and learning rate
        print("loss_at1000", loss_at1000)
        batch_loss_at_1000[pdb_ids[i]] = [loss_at1000[pdb_ids[i]][0], loss_at1000[pdb_ids[i]][1]]
        batch_loss_at_2000[pdb_ids[i]] = [loss_at2000[pdb_ids[i]][0], loss_at2000[pdb_ids[i]][1]]
        batch_loss_at_4000[pdb_ids[i]] = [loss_at4000[pdb_ids[i]][0], loss_at4000[pdb_ids[i]][1]]
        #set up loss per epoch plot and RMSD function then call rmsd optimizer over x epochs

        #save pdb and loss per x epochs then repeat as many times as desired

        #log4 after angles, loss, and epochs

        #save loss vs epoch plot
        loss_per_epoch_plot(loss, epochs, pdb_ids[i])

        #set up final path to save pdb and then write new pdb
        write_pdbs(pdb_ids[i], coords_2, chainnames, resnames, resnums, atomnames, num_atoms)

    #save new csv with all the info in the old csv plus loss at x,y, and z epochs, and (max deviation, ...)?
    loss_1000_df = pd.DataFrame.from_dict(batch_loss_at_1000, 'index', columns=['pdb_id', 'loss at 1000'])
    print(loss_1000_df)
    pdb_df.columns = ['0', '1', 'pdb_id', 'chain name', 'polymer type', 'resolution', 'model', 'length', '1rst res', 'last_res']
    merged_df_1000 = pd.merge(pdb_df, loss_1000_df, how='inner', on='pdb_id')

    loss_2000_df = pd.DataFrame.from_dict(batch_loss_at_2000, 'index', columns=['pdb_id', 'loss at 2000'])
    print(loss_2000_df)
    merged_df_2000 = pd.merge(merged_df_1000, loss_2000_df, how='inner', on='pdb_id')

    loss_4000_df = pd.DataFrame.from_dict(batch_loss_at_4000, 'index', columns=['pdb_id', 'loss at 4000'])
    print(loss_4000_df)
    merged_df_4000 = pd.merge(merged_df_2000, loss_4000_df, how='inner', on='pdb_id')

    print(merged_df_4000)

    merged_df_4000.to_csv('/u2/home_u2/fam95/Downloads/RNA/Batch4/na_pdb_data_and_loss_after_optim.csv')

    end_time = time.time()
    total_time = (end_time - startTime)
    print()
    print("total_time to run script on full pdb files", total_time)