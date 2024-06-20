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

class TestCoords2ElecDefault(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	msg = "Testing electrostatics on a single atom vs default delphi"
	delphiPath = "/home/lupoglaz/Programs/Delphi/Delphicpp_Linux/delphi_v77/bin/delphicpp_release"
	# delphiPath = "/home/talatnt/Projects/Research/Delphi/Delphicpp_v8.4.2_Linux/Release/delphicpp_release"
	delphiParams = {
		"input_files": {
			"pdb": "born.pdb",
			"siz": "born.siz",
			"crg": "born.crg",
		},
		"output_files": {
			"phi": "phimap.cube",
			"eps": "epsmap.cube",
		},
		"float_params": {
			"scale": 2.0,
			"indi": 2.0,
			"exdi": 80.0,
			"prbrad": 0.0,
			"ionrad": 0.0,
			"salt": 0.1,
			"sigma": 1.0,
			"maxc": 0.0001
		},
		"int_params": {
			"gsize": 61,
			"bndcon": 1,
			"nonit": 1,
			"gaussian": 1,
			"linit": 800,
		}
	}

	# pdb_file = """ATOM      1  N   VAL D  10       0.000   0.000   0.000"""
	pdb_file = """ATOM      1  N   VAL D  10       -5.000   0.000   0.000
ATOM      2  N   VAL D  10       5.000   0.000   0.000"""
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
				fout.write('out(%s,file="%s",format=cube)\n'%(key, params["output_files"][key]))
			for key in params["float_params"]:
				fout.write("%s=%f\n" % (key, params["float_params"][key]))
			for key in params["int_params"]:
				fout.write("%s=%d\n" % (key, params["int_params"][key]))

		os.system(self.delphiPath + " params.txt")

	def setUp(self):

		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()

		self.p2c = PDB2CoordsUnordered()
		self.get_center = Coords2Center()
		self.elec_params = ElectrostaticParameters('single_atom', type='born')
		self.a2p = AtomNames2Params()


	def compare_electrostatics(self, delphi_params):
		self.writeFiles(params=delphi_params)
		self.runDelphi(params=delphi_params)

		delphi_phi, spatial_dim, res = cube2numpy(file_path="phimap.cube")
		delphi_eps, spatial_dim, res = cube2numpy(file_path="epsmap.cube")
		
		spatial_dim = delphi_params['int_params']['gsize']
		res = 1.0/delphi_params['float_params']['scale']
		
		box_size = res*spatial_dim
		box_center = torch.tensor([[box_size/2.0, box_size/2.0, box_size/2.0]], dtype=torch.double, device='cpu')

		self.c2e = Coords2Elec(	box_size=spatial_dim,
								resolution=res,
								eps_in=2.0,
								eps_out=80.0,
								ion_size=delphi_params['float_params']['ionrad'],
								wat_size=delphi_params['float_params']['prbrad'],
								asigma=delphi_params['float_params']['sigma'],
								kappa02=8.48*delphi_params['float_params']['salt'],
								charge_conv=7046.52,
								d=13)

		prot = self.p2c([delphi_params["input_files"]["pdb"]])
		prot_center = self.get_center(prot[0], prot[-1])
		coords_ce = self.translate(prot[0], -prot_center + box_center, prot[-1])

		params = self.a2p(prot[2], prot[4], prot[-1], self.elec_params.types, self.elec_params.params)
		q, this_eps, this_phi = self.c2e(	coords_ce.to(device='cuda', dtype=torch.float),
								params.to(device='cuda', dtype=torch.float),
								prot[-1].to(device='cuda'))

		delphi_phi = torch.from_numpy(delphi_phi).clamp(0.0, 100.0)
		this_phi = this_phi[0, :, :, :].clamp(0.0, 100.0).to(device='cpu')
		av_err_phi = torch.mean(torch.abs(delphi_phi - this_phi))

		this_eps = this_eps[0, 0, :, :, :].to(device='cpu')
		delphi_eps = torch.from_numpy(delphi_eps)
		av_err_eps = torch.mean(torch.abs(delphi_eps - this_eps))

		this_q = q[0,:,:,:].sum(dim=2).sum(dim=1).to(device='cpu')
				
		return av_err_phi, av_err_eps, delphi_phi.numpy(), delphi_eps.numpy(), this_phi.numpy(), this_eps.numpy(), this_q.numpy()


	def runTest(self):
		
		#Default parameters:
		err_phi, err_eps, delphi_phi, delphi_eps, this_phi, this_eps, this_q = self.compare_electrostatics(self.delphiParams)
		print('Phi error:', err_phi, 'Eps error:', err_eps)
		
		spatial_dim = self.delphiParams['int_params']['gsize']
		
		f = plt.figure()
		plt.subplot(131, title=r'$\phi$')
		plt.plot(this_phi[:, int(spatial_dim/2), int(spatial_dim/2)], label='Our algorithm')
		plt.plot(delphi_phi[:, int(spatial_dim/2), int(spatial_dim/2)], label='Delphi')
		plt.legend(loc="best")

		plt.subplot(132, title=r'$\epsilon$')
		plt.plot(this_eps[:, int(spatial_dim/2), int(spatial_dim/2)], label='Our algorithm')
		plt.plot(delphi_eps[:, int(spatial_dim/2), int(spatial_dim/2)], label='Delphi')
		plt.legend()

		plt.subplot(133, title=r'$Q$')
		plt.plot(this_q, label='Charge')
		plt.legend()
		plt.savefig(os.path.join('TestFig', 'default.png'))
		

# class TestCoords2ElecParam(TestCoords2ElecDefault):
# 	msg = "Testing electrostatics on a single atom using different parameters vs delphi"
# 	paramVariation = {
# 		"float_params": {
# 			"scale": 2.0,
# 			"indi": 2.0,
# 			"exdi": 80.0,
# 			"prbrad": [ x/5.0 for x in range(10)],
# 			"ionrad": [ x/5.0 for x in range(10)],
# 			"salt": [ x/2.0 for x in range(20)],
# 			"sigma": 1.0,
# 			"maxc": 0.0001
# 		}
# 	}
# 	def runTest(self):
		
# 		#Default parameters:
# 		for param_type in self.paramVariation.keys():
# 			for param_name in self.paramVariation[param_type].keys():
# 				if isinstance(self.paramVariation[param_type][param_name], list):
# 					f = plt.figure()
# 					plt.title(param_name)
# 					errs_phi = []
# 					errs_eps = []
# 					for param in self.paramVariation[param_type][param_name]:
# 						var_param = self.delphiParams.copy()
# 						var_param[param_type][param_name] = param
# 						err_phi, err_eps, delphi_phi, delphi_eps, this_phi, this_eps = self.compare_electrostatics(var_param)
# 						errs_eps.append(err_eps)
# 						errs_phi.append(err_phi)
# 					plt.plot(self.paramVariation[param_type][param_name], errs_phi, label = 'Errors phi')
# 					plt.plot(self.paramVariation[param_type][param_name], errs_eps, label = 'Errors eps')
# 					plt.legend(loc="best")
# 					plt.savefig(os.path.join('TestFig', param_name+'.png'))
# 					# plt.show()


if __name__ == '__main__':
	unittest.main()