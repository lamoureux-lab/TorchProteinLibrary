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


class TestAngles2BackboneJacobian(TestAngles2Backbone):
	
	def compute_jacobian(f, x, output_dims):
		'''
		Normal:
			f: input_dims -> output_dims
		Jacobian mode:
			f: output_dims x input_dims -> output_dims x output_dims
		'''
		repeat_dims = tuple(output_dims) + (1,) * len(x.shape)
		jac_x = x.detach().repeat(*repeat_dims)
		jac_x.requires_grad_()
		jac_y = f(jac_x)
		
		ml = torch.meshgrid([torch.arange(dim) for dim in output_dims])
		index = [m.flatten() for m in ml]
		gradient = torch.zeros(output_dims + output_dims)
		gradient.__setitem__(tuple(index)*2, 1)
		
		jac_y.backward(gradient)
			
		return jac_x.grad.data

	def runTest(self):
		from torch.autograd.gradcheck import get_analytical_jacobian
		import matplotlib.pyplot as plt
		length = 120
		batch_size = 1
		device = 'cuda'
		# device = 'cpu'
		backbone_angles = torch.randn(batch_size, 3, length, dtype=torch.float, device=device).requires_grad_()
		num_aa = torch.zeros(batch_size, dtype=torch.int, device=device).random_(int(length/2), length)		
		output = self.a2b(backbone_angles, num_aa)

		Jacobian = torch.zeros(3, length, 3*3*length, dtype=torch.float, device=device)
		for i in range(3*3*length):
			L = torch.zeros_like(output)
			L[0, i] = 1.0
			output.backward(L, retain_graph=True)
			Jacobian[:, :, i] = backbone_angles.grad
			backbone_angles.grad.fill_(0.0)

		Jacobian = Jacobian.view(3*length, 3*3*length)
		U, Sigma, V = Jacobian.svd()
		plt.subplot(2, 1, 1)
		plt.imshow(Jacobian[:,:].cpu().numpy())
		plt.colorbar()

		plt.subplot(2, 1, 2)
		plt.plot(Sigma.cpu().numpy())
		plt.show()
		
		

		
if __name__=='__main__':
	unittest.main()