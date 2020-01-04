import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, writePDB
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters, Coords2Elec
import _Volume
from read_cube import cube2numpy # reads .cube Delphi's output and converts to np array

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

from TorchProteinLibrary.Utils import VolumeField, ProteinStructure
import vtkplotter as vtkplotter

# import pyvista as pv
import matplotlib.pylab as plt

# from test import TestCoords2EpsSingleAtom

class TestCoords2EpsUBI(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	msg = "Testing electrostatics on ubiquitin"
	delphiPath = "/home/lupoglaz/Programs/Delphi/Delphicpp_Linux/delphi_v77/bin/delphicpp_release"
	delphiParams = {
		"input_files": {
			"pdb": "tmp.pdb",
			"siz": "tmp.siz",
			"crg": "tmp.crg",
		},
		"output_files": {
			"phi": "phimap.cube",
			"eps": "epsmap.cube",
		},
		"float_params": {
			"scale": 0.66,
			"indi": 2.0,
			"exdi": 80.0,
			"prbrad": 0.0,
			"ionrad": 0.0,
			"salt": 0.1,
			"sigma": 1.0,
			"maxc": 0.0001
		},
		"int_params": {
			"gsize": 80,
			"bndcon": 1,
			"nonit": 1,
			"gaussian": 1,
			"linit": 800,
		}
	}

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
		delphi_phi, spatial_dim, res = cube2numpy(file_path="phimap.cube")
		delphi_phi = torch.from_numpy(delphi_phi).clamp(-300.0, 300.0).numpy()
		delphi_eps, spatial_dim, res = cube2numpy(file_path="epsmap.cube")
		
		return delphi_phi, delphi_eps

	def setUp(self):
		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()
		self.box_size = self.delphiParams['int_params']['gsize']
		self.res = 1.0/self.delphiParams['float_params']['scale']

		self.c2e = Coords2Elec(box_size=self.box_size, resolution=self.res)
		self.p2c = PDB2CoordsUnordered()
		self.get_center = Coords2Center()
		self.elec_params = ElectrostaticParameters('protein', type='amber')
		self.a2p = AtomNames2Params()
		self.elec_params.saveTinker('tmp')

	def runTest(self):
		box_center = torch.tensor([[1.0/2.0, 1.0/2.0, 1.0/2.0]], dtype=torch.double, device='cpu') * self.res * self.box_size

		self.c2e = Coords2Elec(	box_size=self.box_size,
								resolution=self.res,
								eps_in=2.0,
								eps_out=80.0,
								ion_size=self.delphiParams['float_params']['ionrad'],
								wat_size=self.delphiParams['float_params']['prbrad'],
								asigma=self.delphiParams['float_params']['sigma'],
								kappa02=8.48*self.delphiParams['float_params']['salt'],
								charge_conv=7046.52,
								d=3)

		prot = self.p2c(["data/1brs.pdb"])
		prot_center = self.get_center(prot[0], prot[-1])
		coords_ce = self.translate(prot[0], -prot_center + box_center, prot[-1])

		writePDB('tmp.pdb', coords_ce, prot[1], prot[2], prot[3], prot[4], prot[5])

		params = self.a2p(prot[2], prot[4], prot[-1], self.elec_params.types, self.elec_params.params)
		q, eps, phi = self.c2e(	coords_ce.to(device='cuda', dtype=torch.float),
								params.to(device='cuda', dtype=torch.float),
								prot[-1].to(device='cuda'))
		
		
		this_eps = eps[0, 0, :, :, :].to(device='cpu').numpy()
		this_phi = phi[0, :, :, :].to(device='cpu').clamp(-300.0, 300.0).numpy()

		delphi_phi, delphi_eps = self.runDelphi(self.delphiParams)

		f = plt.figure()
		plt.subplot(121, title=r'$\phi$')
		plt.plot(this_phi[:, int(self.box_size/2), int(self.box_size/2)], label='Our algorithm')
		plt.plot(delphi_phi[:, int(self.box_size/2), int(self.box_size/2)], label='Delphi')
		plt.legend(loc="best")

		plt.subplot(122, title=r'$\epsilon$')
		plt.plot(this_eps[:, int(self.box_size/2), int(self.box_size/2)], label='Our algorithm')
		plt.plot(delphi_eps[:, int(self.box_size/2), int(self.box_size/2)], label='Delphi')
		plt.legend()
		# plt.show()

		coords, chains, resnames, resnums, atomnames, numatoms = prot

		p2c = PDB2CoordsUnordered()
		prot = ProteinStructure(coords_ce, chains, resnames, resnums, atomnames, numatoms)
		atoms_plot = prot.plot_atoms()
		prot = ProteinStructure(*prot.select_CA())
		backbone_plot = prot.plot_tube()
		
		delphi_phi = torch.from_numpy(delphi_phi).unsqueeze(dim=0)
		delphi_eps = torch.from_numpy(delphi_eps).unsqueeze(dim=0)
		
		vp = vtkplotter.Plotter(N=4, title='basic shapes', axes=0)
		vp.sharecam = True 
		
		this_iso = VolumeField(phi).get_actor(threshold=[-2,2])
		delphi_iso = VolumeField(delphi_phi).get_actor(threshold=[-2,2])
		this_eps = VolumeField(eps[:,0,:,:,:]).get_actor(threshold=10)
		delphi_eps = VolumeField(delphi_phi).get_actor(threshold=10)

		vp.show(this_iso, vtkplotter.Text('this phi'), at=0)
		vp.show(delphi_iso, vtkplotter.Text('delphi phi'), at=1)
		vp.show(this_eps, vtkplotter.Text('this eps'), at=2)
		vp.show(delphi_eps, vtkplotter.Text('delphi eps'), at=3, interactive=1)
		# p = VtkPlotter()
		# p.add(atoms_plot)
		# p.add(backbone_plot)
		# p.add(VolumeField(phi.to(device='cpu'), resolution=self.res).plot_scalar(contour_value=1.0))
		# # p.add(VolumeField(delphi_phi, resolution=self.res).plot_scalar(contour_value=2.5))
		# p.show()

if __name__ == '__main__':
	unittest.main()