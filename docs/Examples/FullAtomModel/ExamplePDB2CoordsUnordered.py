import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3

from scipy.spatial import ConvexHull
cmap = matplotlib.cm.get_cmap('Spectral')

if __name__=='__main__':
    
    #Reading pdb file
    p2c = FullAtomModel.PDB2CoordsUnordered()
    coords, chains, res_names, res_nums, atom_names, num_atoms = p2c(["f4TQ1_B.pdb"])
    
    #Resizing coordinates array for convenience
    N = int(num_atoms[0].item())
    coords.resize_(1, N, 3)

    
    N_res = res_nums[0,-1].item()+1
    residue_hulls = []
    residue_coords = []
    for i in range(N_res):
        mask = torch.eq(res_nums[0,:], i)
        num_res_atoms = int(mask.sum().item())

        # Obtaining coordinates of all atoms of a single residue
        coords_mask = torch.stack([mask, mask, mask], dim = 1).unsqueeze(dim=0)
        res_coords = coords.masked_select(coords_mask).resize(num_res_atoms, 3).numpy()
        residue_coords.append(res_coords)
        
        # Constructing a convex hull
        hull = ConvexHull(res_coords)
        residue_hulls.append(hull)
    
    #Plotting all atoms with the red dots
    cmap = matplotlib.cm.get_cmap('Set3')
    sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
    fig = plt.figure()
    plt.title("PDB2CoordsUnordered")
    ax = p3.Axes3D(fig)
    
    i_res = 0
    for hull, coords in zip(residue_hulls,residue_coords):
        for simplex in hull.simplices:
            ax.plot(coords[simplex, 0], coords[simplex,1], coords[simplex,2], color = cmap(float(i_res)/float(N_res)))
        i_res += 1

    ax.plot(sx,sy,sz, 'r.', label = 'atoms')
    ax.legend()
    # plt.show()
    plt.savefig("ExamplePDB2CoordsUnordered.png")