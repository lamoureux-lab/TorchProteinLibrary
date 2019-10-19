import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters, Coords2Elec
from read_cube import cube2numpy # reads .cube Delphi's output and converts to np array

pdb_name = "test"
delphi_out_file_path = "data/"+pdb_name+"_delphi.cube"
pdb_file_path = "data/"+pdb_name+".pdb"

ele_param_path = "data"
force_field_type = "amber"

device = 'cuda'
dtype = torch.float
p2c = PDB2CoordsUnordered()
a2p =  AtomNames2Params()
translate = CoordsTranslate()
get_center = Coords2Center()
elec_params = ElectrostaticParameters(ele_param_path, type = force_field_type) 


#get delphi phi
Delphi, spatial_dim, res = cube2numpy(file_path = delphi_out_file_path)
box_size = spatial_dim*res

unit_factor = 6987.0 #this factor ensures that potential
                     #unit is in (k_B*T / e), where T = 300 K room temp.

c2e = Coords2Elec(box_size = spatial_dim, #must be spatial_dim
                  resolution = res,
                  eps_in = 2.0,
                  eps_out = 79.0,
                  stern_size = 0.0, #1.0,
                  kappa02 = 0.8486,
                  charge_conv = unit_factor) #7046.52 )

box_center = torch.tensor([[box_size/2.0, box_size/2.0, box_size/2.0]], dtype=torch.double, device='cpu')
coords, chain_names, resnames, resnums, anames, num_atoms = p2c([pdb_file_path])
print(coords, num_atoms)
params = a2p(resnames, anames, num_atoms, elec_params.types, elec_params.params)
center = get_center(coords, num_atoms)
ce_coords = translate(coords, -center + box_center, num_atoms)

#get q, eps, phi
q, eps, phi = c2e(ce_coords.to(device = device, dtype = dtype), 
		params.to(device = device, dtype = dtype), 
		num_atoms.to(device=device))

pot = phi[0,:,:,:].to(device='cpu').numpy()

diff_phi = pot - Delphi
print("RMS Difference",np.sqrt((diff_phi**2).sum() / pot.shape[0]**3))

import pyvista as pv
plot_title = r'$\phi - \phi_{Delphi}$'
p = pv.Plotter(point_smoothing=True)
p.add_text(plot_title, position='upper_left', font_size=18, color=None)
p.add_volume(np.abs(diff_phi), cmap="viridis", opacity="linear")
p.show()
