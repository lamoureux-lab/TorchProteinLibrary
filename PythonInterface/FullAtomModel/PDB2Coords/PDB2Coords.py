# TO DO:
# 1. pdb2sequence
# 2. loading and allocating array
# 3. dealing with missing aa

from Exposed import cppPDB2Coords
import torch


class PDB2Coords:
    def __init__(self, filename):
        self.filename = filename
        
