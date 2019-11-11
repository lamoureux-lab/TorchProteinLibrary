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

# from test import TestCoords2EpsSingleAtom

class TestCoords2EpsUBI(unittest.TestCase):
	device = 'cuda'
	dtype = torch.float
	msg = "Testing electrostatics on ubiquitin"
	pdb_file = ""
	siz_file = ""
	crg_file = ""
	def writeFiles(self, params):
		pass

	def setUp(self):
		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()
		self.box_size = 80
		self.res = 1.5

		self.c2e = Coords2Elec(box_size=self.box_size, resolution=self.res)
		self.p2c = PDB2CoordsUnordered()
		self.get_center = Coords2Center()
		self.elec_params = ElectrostaticParameters('protein', type='amber')
		self.a2p = AtomNames2Params()

	def runTest(self):
		box_center = torch.tensor([[1.0/2.0, 1.0/2.0, 1.0/2.0]], dtype=torch.double, device='cpu') * self.res * self.box_size

		self.c2e = Coords2Elec(	box_size=self.box_size,
								resolution=self.res,
								eps_in=2.0,
								eps_out=80.0,
								ion_size=0.0,
								wat_size=0.0,
								asigma=1.1,
								kappa02=0.0,
								charge_conv=7046.52,
								d=7)

		prot = self.p2c(["protein/1brs.pdb"])
		prot_center = self.get_center(prot[0], prot[-1])
		coords_ce = self.translate(prot[0], -prot_center + box_center, prot[-1])

		params = self.a2p(prot[2], prot[4], prot[-1], self.elec_params.types, self.elec_params.params)
		# with torch.autograd.profiler.profile(use_cuda=True) as prof:
		q, eps, phi = self.c2e(	coords_ce.to(device='cuda', dtype=torch.float),
								params.to(device='cuda', dtype=torch.float),
								prot[-1].to(device='cuda'))
		# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
		
		this_eps = eps[0, 0, :, :, :].to(device='cpu').numpy()
		this_phi = phi[0, :, :, :].to(device='cpu').clamp(-300.0, 300.0).numpy()

		p = pv.Plotter(point_smoothing=True)
		# p.add_volume(Delphi, cmap="viridis", opacity="linear")
		p.add_volume(np.abs(this_phi), cmap="viridis", opacity="linear")
		p.show()

		f = plt.figure()
		plt.subplot(121, title=r'$\phi$')
		plt.plot(this_phi[:, int(self.box_size/2), int(self.box_size/2)], label='Our algorithm')
		# plt.plot(delphi_phi[:, int(spatial_dim/2), int(spatial_dim/2)], label='Delphi')
		plt.legend(loc="best")

		plt.subplot(122, title=r'$\epsilon$')
		plt.plot(this_eps[:, int(self.box_size/2), int(self.box_size/2)], label='Our algorithm')
		# plt.plot(Eps[:, int(spatial_dim/2), int(spatial_dim/2)], label='Delphi')
		plt.legend()
		plt.show()

if __name__ == '__main__':
	unittest.main()