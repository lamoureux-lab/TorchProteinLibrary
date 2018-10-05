import sys
import os
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea

import torch
from TorchProteinLibrary import FullAtomModel

if __name__=='__main__':

    # p2c = FullAtomModel.PDB2Coords.PDB2CoordsBiopython()
    p2c = FullAtomModel.PDB2CoordsUnordered()
    coords, res, anames, num_atoms = p2c(["f4TQ1_B.pdb"])
    print (coords.size())
    print (res.size())
    print (anames.size())
    print (num_atoms)

    coords = coords.numpy()
    coords = coords.reshape(int(coords.shape[1]/3), 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    ax.scatter(x,y,z)
    plt.show()
