import sys
import os
import torch

from PDB2Volume import PDB2Volume

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from Visualization import VisualizeVolume4d

if __name__=='__main__':
    pdb2v = PDB2Volume()
    pdb2v.cuda()
    volume = pdb2v("/media/lupoglaz/ProteinsDataset/CASP12_SCWRL/T0859/IntFOLD4_TS1")
    print torch.sum(volume)
    # VisualizeVolume4d(volume)
    