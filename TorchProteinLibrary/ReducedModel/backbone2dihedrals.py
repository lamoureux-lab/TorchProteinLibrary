import torch
from Bio.PDB import calc_angle, rotaxis, Vector, calc_dihedral

def backbone2dihedrals(backbone, length):
    batch_size = backbone.size(0)
    max_seq_length = int(backbone.size(1)/9)
    backbone = backbone.view(batch_size, max_seq_length*3, 3)
    dihedrals = torch.zeros(batch_size, 3, max_seq_length, dtype=backbone.dtype, device=backbone.device)
    
    for batch_idx in range(batch_size):
        phi = [0.0]
        psi = []
        omega = []
        for i in range(length[batch_idx].item()):
            N_i = Vector(backbone[batch_idx, 3*i+0, :].numpy())
            CA_i = Vector(backbone[batch_idx, 3*i+1, :].numpy())
            C_i = Vector(backbone[batch_idx, 3*i+2,:].numpy())
            
            if i>0:
                C_im1 = Vector(backbone[batch_idx, 3*i-1, :].numpy())
                phi.append(calc_dihedral(C_im1, N_i, CA_i, C_i))
            
            if i<(length[batch_idx].item()-1):
                N_ip1 = Vector(backbone[batch_idx, 3*(i+1), :].numpy())
                psi.append(calc_dihedral(N_i, CA_i, C_i, N_ip1))
            
            if i<(length[batch_idx].item()-1):
                CA_ip1 = Vector(backbone[batch_idx, 3*(i+1)+1, :].numpy())
                omega.append(calc_dihedral(CA_i, C_i, N_ip1, CA_ip1))
        
        psi.append(0.0)
        omega.append(0.0)
        
        dihedrals[batch_idx, :, :] = torch.tensor([phi, psi, omega], dtype=backbone.dtype, device=backbone.device)
    
    return dihedrals