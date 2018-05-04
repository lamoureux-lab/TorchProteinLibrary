import sys
import os
import torch

from PDB2Volume import PDB2Volume, PDB2VolumeLocal

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from Visualization import VisualizeVolume4d

if __name__=='__main__':
    pdb2v = PDB2VolumeLocal()
    # pdb2v.cuda()
    volume, coords, num_atoms = pdb2v(["/media/lupoglaz/ProteinsDataset/CASP12_SCWRL/T0859/IntFOLD4_TS1",
                                        "/media/lupoglaz/ProteinsDataset/CASP12_SCWRL/T0859/IntFOLD4_TS2"])
    print torch.sum(volume)
    print num_atoms
    print coords.size()
    # VisualizeVolume4d(volume)
    