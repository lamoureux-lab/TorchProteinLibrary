import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3


if __name__=='__main__':
    
    #Reading pdb file
    p2c = FullAtomModel.PDB2CoordsUnordered()
    coords, res_names, atom_names, num_atoms = p2c(["f4TQ1_B.pdb"])
    
    #Resizing coordinates array for convenience
    N = int(num_atoms.data[0])
    coords.resize_(1, N, 3)

    #Plotting all atoms with the red dots
    sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
    fig = plt.figure()
    plt.title("Full atom model")
    ax = p3.Axes3D(fig)
    ax.plot(sx,sy,sz, 'r.', label = 'atoms')
    ax.legend()
    plt.savefig("ExamplePDB2CoordsUnordered.png")