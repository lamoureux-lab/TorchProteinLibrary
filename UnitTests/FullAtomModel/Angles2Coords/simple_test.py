import os
import sys
import argparse
import torch
from TorchProteinLibrary import FullAtomModel

if __name__ == "__main__":
    a2c = FullAtomModel.Angles2Coords()
    residues = ["AAAAAAAA"]
    angles = torch.zeros(1, 8, len(residues[0]), dtype=torch.double, device='cpu')
    angles[:, 2] = 3.14
    coords, chain_names, resnames, resnums, anames, num_atoms = a2c(angles, residues)
    c2pdb = FullAtomModel.writePDB("simple_test.pdb", coords, chain_names, resnames, resnums, anames, num_atoms)
    print('done')