import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
import _Physics
import _Volume

p2c = PDB2CoordsUnordered()
translate = CoordsTranslate()
get_center = Coords2Center()

coords, _, _, _, _, num_atoms = p2c(["tmp/1lzy.pdb"])

center = get_center(coords, num_atoms)
coords_ce = translate(coords, -center, num_atoms)
box_center = torch.tensor([[30*2.5/2.0, 30*2.5/2.0, 30*2.5/2.0]], dtype=torch.double)
coords_ce_ce = translate(coords_ce, box_center, num_atoms)

coords_ce_ce = coords_ce_ce.to(dtype=torch.float32)
res = _Physics.test(coords_ce_ce, num_atoms)
print(res.sum())
_Volume.Volume2Xplor(res, "tmp.xplor", 2.5)