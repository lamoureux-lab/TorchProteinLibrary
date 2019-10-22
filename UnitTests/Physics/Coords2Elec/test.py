import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters, Coords2Elec
import _Volume


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

class TestCoords2Eps(TestAtomNames2Params):
	device = 'cuda'
	dtype = torch.float
	msg = "Testing dielectric constant"

	def setUp(self):
		super(TestCoords2Eps, self).setUp()
		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()
		box_size = 60
		res = 0.5
		self.c2e = Coords2Elec(box_size=box_size, resolution=res)
		self.box_center = torch.tensor([[res*box_size/2.0, res*box_size/2.0, res*box_size/2.0]], dtype=torch.double, device='cpu')

class TestCoords2Eps_forward(TestCoords2Eps):

	def runTest(self):
		coords, chain_names, resnames, resnums, anames, num_atoms = self.p2c(["test.pdb"])
		params = self.a2p(resnames, anames, num_atoms, self.elec_params.types, self.elec_params.params)
		center = self.get_center(coords, num_atoms)
		ce_coords = self.translate(coords, -center + self.box_center, num_atoms)
		q, eps, phi = self.c2e(	ce_coords.to(device=self.device, dtype=self.dtype), 
							params.to(device=self.device, dtype=self.dtype), 
							num_atoms.to(device=self.device))
		
		_Volume.Volume2Xplor(phi[0,:,:,:].to(device='cpu'), "tmp.xplor", 1.0)

		import pyvista as pv
		p = pv.Plotter(point_smoothing=True)
		q = torch.abs(q[0,:,:,:]).to(device='cpu').numpy()
		phi = torch.abs(phi[0,:,:,:]).to(device='cpu').numpy()
		# eps = eps[0,:,:,:].to(device='cpu').numpy()
		p.add_volume(phi, cmap="viridis", opacity="linear")
		# p.add_volume(eps, cmap="coolwarm", opacity="linear_r")
		p.show()


class TestCoords2Eps_backward(TestCoords2Eps):

	def runTest(self):
		pass

if __name__ == '__main__':
	unittest.main()

	# # sphinx_gallery_thumbnail_number = 3
	# import pyvista as pv
	# import numpy as np
	# from pyvista import examples
	# # import vtk
	
	# # vtkgrid = vtk.vtkImageData()
	# # print(vtkgrid)
	# grid = pv.UniformGrid((10,10,10))
	# grid.cell_arrays['init'] = np.random.rand(grid.n_cells)
	# print(grid)
	
	# # Volume rendering is not supported with Panel yet
	# pv.rcParams["use_panel"] = False

	# bolt_nut = examples.download_bolt_nut()
	# head = examples.download_head()
	# print(bolt_nut.get(0).point_arrays.keys())
	# print(bolt_nut.get(0).cell_arrays.keys())
	# print(bolt_nut.get(0)['SLCImage'].dtype)
	# print(head.point_arrays.keys())
	# print(head.cell_arrays.keys())
	
	# vol = np.random.rand(1000).reshape((10,10,10))

	# grid.plot(scalars=np.random.rand(grid.n_cells))
	# p = pv.Plotter()
	# p.add_volume(vol)
	# p.show()

	



