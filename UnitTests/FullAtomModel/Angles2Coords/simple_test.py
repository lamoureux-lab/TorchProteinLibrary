import os
import sys
import argparse
import torch
from TorchProteinLibrary import FullAtomModel

if __name__=="__main__":
    a2c = FullAtomModel.Angles2Coords()
    residues = ["AAAAAAAA"]
    angles = torch.zeros(1, 8, len(residues[0]), dtype=torch.double, device='cpu')
    angles[:,2] = 3.14
    protein, res_names, atom_names, num_atoms = a2c(angles, residues)