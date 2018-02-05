import sys
import os
import torch

from PDB2Volume import PDB2Volume

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Visualization import VisualizeVolume4d

if __name__=='__main__':
    pdb2v = PDB2Volume()
    volume = pdb2v("TestData/2lzm.pdb")
    VisualizeVolume4d(volume)
    