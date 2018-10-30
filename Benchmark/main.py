import torch
from torch.autograd import profiler
from TorchProteinLibrary import ReducedModel, FullAtomModel, RMSD
import numpy as np

if __name__=='__main__':
    a2c = FullAtomModel.Angles2Coords()
    sequences = ['GGMLGWAHFGY']
    
    #Setting conformation to alpha-helix
    angles = torch.zeros(len(sequences), 7, len(sequences[-1]), dtype=torch.double, device='cpu')
    angles.data[:,0,:] = -1.047
    angles.data[:,1,:] = -0.698
    angles.data[:,2:,:] = 110.4*np.pi/180.0

    #Converting angles to coordinates
    with profiler.profile() as prof:
        coords, res_names, atom_names, num_atoms = a2c(angles, sequences)
        # coords.backward()
    dur = 0.0
    for evt in prof.function_events:
        dur += evt.cpu_interval.elapsed_us()
    print(dur/1000.0, 'ms')
    
    # print(prof)
    # print(prof.total_average())
