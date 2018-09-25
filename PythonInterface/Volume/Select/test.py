import sys
import os
import torch
import numpy as np
from SelectVolume import SelectVolume

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PDB2Volume import PDB2VolumeLocal


if __name__=='__main__':
	pdb2v = PDB2VolumeLocal()
	volume, coords, num_atoms = pdb2v(["/media/lupoglaz/ProteinsDataset/CASP12_SCWRL/T0859/IntFOLD4_TS1",
										"/media/lupoglaz/ProteinsDataset/CASP12_SCWRL/T0859/IntFOLD4_TS2"])
	sv = SelectVolume()
	features = sv(volume, coords, num_atoms)
	
	batch_idx = 0
	feature_idx = 1
	res = 1.0
	num_max_atoms = coords.size(1)/3
	err = 0.0
	for i in range(0,num_max_atoms):
		x = np.floor(coords[batch_idx,3*i]/res)
		y = np.floor(coords[batch_idx,3*i+1]/res)
		z = np.floor(coords[batch_idx,3*i+2]/res)
		err += np.abs(features[batch_idx, feature_idx, i] - volume[batch_idx, feature_idx, x, y, z])
	
	print 'Error = ', err/num_max_atoms