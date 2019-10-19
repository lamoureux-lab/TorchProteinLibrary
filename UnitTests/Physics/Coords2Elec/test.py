import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters, Coords2Elec
import _Volume
from read_cube import cube2numpy # reads .cube Delphi's output and converts to np array


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

import pyvista as pv
import matplotlib.pylab as plt

class TestCoords2EpsSingleAtom(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	msg = "Testing dielectric constant on a single atom"
	delphiPath = "/home/lupoglaz/Programs/Delphi/Delphicpp_Linux/delphi_v77/bin/delphicpp_release"
	delphiParams = {
		"input_files":{
			"pdb":"born.pdb",
			"siz":"born.siz",
			"crg":"born.crg",
		},
		"output_files":{
			"phi":"phimap.txt",
			"eps":"epsmap.txt",
		},	
		"float_params":{
			"scale":2.0,
			"indi":2.0,
			"exdi":80.0,
			"prbrad":0.0,
			"ionrad":0.0,
			"salt":0.0,
			"sigma":1.0,
			"maxc":0.0001
		},
		"int_params":{
			"gsize":61,
			"bndcon":1,
			"nonit":0,
			"gaussian":1,
			"linit":800,
		}
	}
	pdb_file = """ATOM      1  N   VAL D  10       0.000   0.000   0.000"""
	siz_file = """atom__res_radius_
N     VAL   3.0          
"""
	crg_file = """atom__resnumbc_charge_
N     VAL      10.000            
"""
	def writeFiles(self, params):
		with open(params["input_files"]["pdb"], 'w') as fout:
			fout.write(self.pdb_file)
		with open(params["input_files"]["siz"], 'w') as fout:
			fout.write(self.siz_file)
		with open(params["input_files"]["crg"], 'w') as fout:
			fout.write(self.crg_file)

	def runDelphi(self, params):
		with open("params.txt", 'w') as fout:
			for key in params["input_files"]:
				fout.write('in(%s,file="%s")\n'%(key, params["input_files"][key]))
			for key in params["output_files"]:
				fout.write('in(%s,file="%s")\n'%(key, params["output_files"][key]))
			for key in params["float_params"]:
				fout.write("%s=%f\n"%(key, params["float_params"][key]))
			for key in params["int_params"]:
				fout.write("%s=%d\n"%(key, params["int_params"][key]))

		os.system(self.delphiPath + " params.txt")

	def setUp(self):
		
		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()
		box_size = 60
		res = 0.5
		
		self.c2e = Coords2Elec(box_size=box_size, resolution=res)
		self.p2c = PDB2CoordsUnordered()
		self.get_center = Coords2Center()
		self.elec_params = ElectrostaticParameters('single_atom', type='born')
		self.a2p = AtomNames2Params()
		

	def runTest(self):
		self.writeFiles(params=self.delphiParams)
		self.runDelphi(params=self.delphiParams)
		Delphi, spatial_dim, res = cube2numpy(file_path = "phimap.txt")
		Eps, spatial_dim, res = cube2numpy(file_path = "epsmap.txt")
		box_size = spatial_dim*res
		box_center = torch.tensor([[box_size/2.0, box_size/2.0, box_size/2.0]], dtype=torch.double, device='cpu')
		
		self.c2e = Coords2Elec(	box_size = spatial_dim, #must be spatial_dim
                  				resolution = res,
                  				eps_in = 2.0,
                  				eps_out = 80.0,
                  				ion_size = 0.0,
								wat_size = 0.0,
								asigma = 1.0,
                  				debye_length = 0.0, #0.8486,
                  				charge_conv = 7046.52,#6987.0,
								d = 7)

		prot = self.p2c(["single_atom/born.pdb"])
		prot_center = self.get_center(prot[0], prot[-1])
		coords_ce = self.translate(prot[0], -prot_center + box_center, prot[-1])

		params = self.a2p(prot[2], prot[4], prot[-1], self.elec_params.types, self.elec_params.params)
		q, eps, phi = self.c2e(	coords_ce.to(device='cuda', dtype=torch.float), 
								params.to(device='cuda', dtype=torch.float),
								prot[-1].to(device='cuda'))
		
		this_eps = eps[0,0,:,:,:].to(device='cpu').numpy()
		this_phi = phi[0,:,:,:].to(device='cpu').numpy()
		
		
		p = pv.Plotter(point_smoothing=True)
		p.add_volume(Delphi, cmap="viridis", opacity="linear")
		p.add_volume(this_phi, cmap="viridis", opacity="linear")
		p.show()

		f = plt.figure()
		plt.subplot(121)
		plt.plot(this_phi[:,int(spatial_dim/2),int(spatial_dim/2)], label='Our algorithm')
		plt.plot(Delphi[:,int(spatial_dim/2),int(spatial_dim/2)], label='Delphi')
		plt.legend()
		plt.subplot(122)
		plt.plot(this_eps[:,int(spatial_dim/2),int(spatial_dim/2)], label='Our algorithm')
		plt.plot(Eps[:,int(spatial_dim/2),int(spatial_dim/2)], label='Delphi')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	unittest.main()