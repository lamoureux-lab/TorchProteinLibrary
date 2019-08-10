import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel.Angles2Coords import Angles2Coords
from TorchProteinLibrary.ReducedModel.Angles2Backbone import Angles2Backbone

class TestAngles2Backbone(unittest.TestCase):
	def _plot_coords(coords, filename):
		if not os.path.exists("TestFig"):
			os.mkdir("TestFig")

		min_xyz = -1.5
		max_xyz = 1.5
		coords = coords.numpy()
		sx, sy, sz = coords[:,0], coords[:,1], coords[:,2]
		fig = plt.figure()
		ax = p3.Axes3D(fig)
		ax.plot(sx, sy, sz, '.', label = plot_name)
		ax.set_xlim(min_xyz,max_xyz)
		ax.set_ylim(min_xyz,max_xyz)
		ax.set_zlim(min_xyz,max_xyz)
		ax.legend()
		plt.savefig('TestFig/%s'%filename)

	def setUp(self):
		self.a2b = Angles2Backbone()
		self.a2c = Angles2Coords()

class TestAngles2BackboneForward(TestAngles2Backbone):
	def runTest(self):
		length = 32
		batch_size = 16
		device = 'cuda'
		# device = 'cpu'
		backbone_angles = torch.randn(batch_size, 3, length, dtype=torch.float, device=device)
		num_aa = torch.zeros(batch_size, dtype=torch.int, device=device).random_(int(length/2), length)
		coords_backbone = self.a2b(backbone_angles, num_aa)
		coords_backbone = coords_backbone.cpu().resize_(batch_size, 3*length, 3).numpy()
		
		sequences = [''.join(['G' for i in range(num_aa[j].item())]) for j in range(batch_size)]
		fa_angles = torch.zeros(batch_size, 8, length, dtype=torch.float)
		fa_angles[:,0:3,:].copy_(backbone_angles[:,0:3,:].cpu())
		
		coords_fa, res_names, atom_names, num_atoms = self.a2c(fa_angles, sequences)
		max_num_atoms = torch.max(num_atoms)
		coords_fa = coords_fa.cpu().resize_(batch_size, max_num_atoms, 3).numpy()

		error = 0.0
		N=0
		for j in range(batch_size):
			
			k=0
			for i in range(num_atoms[j].item()):
				compute = False
				if atom_names[j,i,0] == 67 and  atom_names[j,i,1] == 0: #C
					compute = True
				if atom_names[j,i,0] == 67 and  atom_names[j,i,1] == 65 and  atom_names[j,i,2] == 0: #CA
					compute = True
				if atom_names[j,i,0] == 78 and  atom_names[j,i,1] == 0: #N
					compute = True

				if compute:
					E = np.linalg.norm(coords_backbone[j,k,:] - coords_fa[j,i,:])
					error += E
					k+=1
			N+=k
		self.assertAlmostEqual(error/float(N), 0.0, places=5)

class TestAngles2BackboneBackward(TestAngles2Backbone):
	def runTest(self):
		length = 8
		batch_size = 8
		device = 'cuda'
		# device = 'cpu'
		backbone_angles = torch.randn(batch_size, 3, length, dtype=torch.float, device=device).requires_grad_()
		num_aa = torch.zeros(batch_size, dtype=torch.int, device=device).random_(int(length/2), length)
				
		result = torch.autograd.gradcheck(self.a2b, (backbone_angles, num_aa), eps=1e-3, atol=1e-2, rtol=0.01)
		self.assertTrue(result)
		
if __name__=='__main__':
	unittest.main()