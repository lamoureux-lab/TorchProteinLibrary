from core import ModuleBenchmark
import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD
import numpy as np
from Bio.PDB.Polypeptide import aa1
import random

def gen_rand_seq(seq_len=150):
    return ''.join([random.choice(aa1) for i in range(seq_len)])

class Angles2CoordsBenchmark(ModuleBenchmark):
    def __init__(self, device='cpu', num_sequences=32, seq_length= 350):
        self.module = FullAtomModel.Angles2Coords()
        self.sequences = [gen_rand_seq(seq_length) for i in range(num_sequences)]
        print(self.sequences)
        self.angles = torch.randn(len(self.sequences), 7, len(self.sequences[-1]), dtype=torch.double, device='cpu', requires_grad=True)
        
    def prepare(self):
        self.angles.requires_grad_()
        self.coords, res_names, atom_names, num_atoms = self.module(self.angles, self.sequences)
        self.grad_coords = torch.ones(self.coords.size(), dtype=torch.double, device='cpu')
    
    def run_forward(self):
        self.angles.detach()
        self.coords, res_names, atom_names, num_atoms = self.module(self.angles, self.sequences)

    def run_backward(self):
        self.coords.backward(self.grad_coords)

if __name__=='__main__':
    mb = Angles2CoordsBenchmark()
    dur = mb.measure_forward()
    print('Forward:', dur/1000.0, 'ms')
    dur = mb.measure_backward()
    print('Backward:', dur/1000.0, 'ms')