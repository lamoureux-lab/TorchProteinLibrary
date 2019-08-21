import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3


if __name__=='__main__':
    a2c = FullAtomModel.Angles2Coords()
    sequences = ['GGMLGWAHFGY']
    
    #Setting conformation to alpha-helix
    angles = torch.zeros(len(sequences), 8, len(sequences[-1]), dtype=torch.double, device='cpu')
    angles.data[:,0,:] = -1.047
    angles.data[:,1,:] = -0.698
    angles.data[:,3,:] = np.pi
    angles.data[:,3:,:] = 110.4*np.pi/180.0

    #Converting angles to coordinates
    coords, res_names, atom_names, num_atoms = a2c(angles, sequences)
    
    #Making a mask on CA, C, N atoms
    is0C = torch.eq(atom_names[:,:,0], 67).squeeze()
    is1A = torch.eq(atom_names[:,:,1], 65).squeeze()
    is20 = torch.eq(atom_names[:,:,2], 0).squeeze()
    is0N = torch.eq(atom_names[:,:,0], 78).squeeze()
    is10 = torch.eq(atom_names[:,:,1], 0).squeeze()
    isCA = is0C*is1A*is20
    isC = is0C*is10
    isN = is0N*is10
    isSelected = torch.ge(isCA + isC + isN, 1)

    #Resizing coordinates array for convenience (to match selection mask)
    N = int(num_atoms[0].item())
    coords.resize_(1, N, 3)
    
    backbone_x = torch.masked_select(coords[0,:,0], isSelected)
    backbone_y = torch.masked_select(coords[0,:,1], isSelected)
    backbone_z = torch.masked_select(coords[0,:,2], isSelected)

    #Plotting all atoms with the red dots and backbone with the blue line
    sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
    bx, by, bz = backbone_x.numpy(), backbone_y.numpy(), backbone_z.numpy()
    
    fig = plt.figure()
    plt.title("Full atom model")
    ax = p3.Axes3D(fig)
    ax.plot(sx,sy,sz, 'r.', label = 'atoms')
    ax.plot(bx,by,bz, 'b-', label = 'backbone')
    ax.legend()
    plt.show()
    # plt.savefig("ExampleAngles2Coords.png")