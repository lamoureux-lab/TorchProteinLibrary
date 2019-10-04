import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
import _Physics
p2c = PDB2CoordsUnordered()
coords, _, _, _, _, num_atoms = p2c(["tmp/1lzy.pdb"])
print(_Physics.test(coords, num_atoms))