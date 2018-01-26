import sys
import os
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim

from PDB2Coords.Exposed import cppPDB2Coords
from Coords2RMSD.Coords2RMSD import Coords2RMSD
from Angles2Coords.Angles2Coords import Angles2Coords

if __name__=='__main__':
	sequence = 'MNIFEMLRIDERLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSIWYNQTPNRAKRVITTFRTGTWDAYKNL'
	num_atoms = cppPDB2Coords.getSeqNumAtoms(sequence)
	num_angles = 7
	
	target_coords = Variable(torch.DoubleTensor(3*num_atoms))
	cppPDB2Coords.PDB2Coords("PDB2Coords/TestFig/2lzm.pdb", target_coords.data)
	
	angles = Variable(torch.DoubleTensor(num_angles, len(sequence)).zero_(), requires_grad=True)
	loss = Coords2RMSD(num_atoms)
	a2c = Angles2Coords(sequence, num_atoms)
	
	v_num_atoms = Variable(torch.IntTensor(1).fill_(num_atoms))


	optimizer = optim.Adam([angles], lr = 0.001)
	for i in range(0,1000):
		coords = a2c(angles)
		rmsd = loss(coords, target_coords, v_num_atoms)
		rmsd_real = np.sqrt(rmsd.data[0])
		print rmsd_real
		rmsd.backward()
		optimizer.step()