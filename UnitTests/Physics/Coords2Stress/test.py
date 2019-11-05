import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import CoordsTranslate, Coords2Center
from TorchProteinLibrary.Physics import AtomNames2Params, Coords2Stress
import _Volume

import prody as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from AtomNames2Params import TestAtomNames2Params

import matplotlib.pylab as plt
import pyvista as pv

class TestCoords2Stress(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 7
	batch_size = 16
	max_num_atoms = 30
	eps=1e-06 
	atol=1e-05 
	rtol=0.001
	msg = "Testing stress tensor computation"

	def setUp(self):
		num_at = 10
		self.box_size = 80
		self.resolution = 1.5
		self.translate = CoordsTranslate()
		self.get_center = Coords2Center()
		self.coords = torch.randn(1, num_at*3, dtype=torch.float, device=self.device)
		self.num_atoms = torch.tensor([num_at], dtype=torch.int, device=self.device)
		for i in range(num_at):
			self.coords[0, 3*i + 0] = i*2.0 + 2.0
			self.coords[0, 3*i + 1] = 0.0#i*1.5 + 2.0
			self.coords[0, 3*i + 2] = 0.0#i*1.5 + 2.0

		self.coords2stress = Coords2Stress(box_size=self.box_size, resolution=self.resolution)
		self.box_center = torch.tensor([[self.box_size/2.0, self.box_size/2.0, self.box_size/2.0]], dtype=torch.float, device='cpu')


class TestCoords2Stress_forward(TestCoords2Stress):

	def runTest(self):
		pdb = pd.parsePDB('1ubi.pdb')
		atoms = pdb.select('protein and calpha')

		self.coords = torch.randn(1, len(atoms)*3, dtype=torch.float, device=self.device)
		self.num_atoms = torch.tensor([len(atoms)], dtype=torch.int, device=self.device)
		
		for i, atom in enumerate(atoms):
			self.coords[0, i*3 + 0] = atom.getCoords()[0]
			self.coords[0, i*3 + 1] = atom.getCoords()[1]
			self.coords[0, i*3 + 2] = atom.getCoords()[2]


		prot_center = self.get_center(self.coords, self.num_atoms)
		coords_ce = self.translate(self.coords, -prot_center + self.box_center, self.num_atoms)

		anm = pd.ANM('p38 ANM analysis')
		anm.buildHessian(atoms)
				
		anm.calcModes(n_modes=1)
		print(anm.getEigvals())
		print(anm.getCovariance())
		
		dist_mat, dr, vol = self.coords2stress(coords_ce, self.num_atoms)
		vol = torch.sqrt((vol*vol).sum(dim=1))
		print(dr)
		
		p = pv.Plotter(point_smoothing=True)
		p.add_volume(torch.abs(vol[0,:,:,:]).cpu().numpy(), cmap="viridis", opacity="linear")
		p.show()
		
		
		f = plt.figure(figsize=(15,5))
		plt.subplot(1,3,1)
		plt.imshow(dist_mat[0,:,:].numpy())
		plt.colorbar()

		plt.subplot(1,3,2)
		plt.imshow(anm.getHessian())
		plt.colorbar()

		plt.subplot(1,3,3)
		plt.imshow(dist_mat[0,:,:].numpy() - anm.getHessian())
		plt.colorbar()

		plt.tight_layout()
		
		plt.show()


if __name__=='__main__':
	unittest.main()