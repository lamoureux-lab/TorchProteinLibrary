import os
import sys
import unittest

import torch
from TorchProteinLibrary import FullAtomModel
from TorchProteinLibrary.FullAtomModel import Angles2Coords, PDB2CoordsUnordered, Coords2Center, CoordsTranslate, CoordsRotate
from TorchProteinLibrary.ReducedModel import Angles2Backbone
from TorchProteinLibrary.RMSD import Coords2RMSD

from torch import optim
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from matplotlib.animation import FuncAnimation


class VisualTestAngles2Backbone(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 7
	batch_size = 1
	eps=1e-06
	atol=1e-05
	rtol=0.001
	msg = "Visualizing Angles2Backbone"

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

	def prepare_structure(self):
		p2c = PDB2CoordsUnordered()
		coords_dst, chain_names, res_names_dst, res_nums_dst, atom_names_dst, num_atoms_dst = p2c(["f4TQ1_B.pdb"])

		#Making a mask on CA, C, N atoms
		is0C = torch.eq(atom_names_dst[:,:,0], 67).squeeze()
		is1A = torch.eq(atom_names_dst[:,:,1], 65).squeeze()
		is20 = torch.eq(atom_names_dst[:,:,2], 0).squeeze()
		is0N = torch.eq(atom_names_dst[:,:,0], 78).squeeze()
		is10 = torch.eq(atom_names_dst[:,:,1], 0).squeeze()
		isCA = is0C*is1A*is20
		isC = is0C*is10
		isN = is0N*is10
		isSelected = torch.ge(isCA + isC + isN, 1)
		num_backbone_atoms = isSelected.sum().item()

		#Resizing coordinates array for convenience
		N = int(num_atoms_dst[0].item())
		coords_dst.resize_(1, N, 3)

		backbone_x = torch.masked_select(coords_dst[0,:,0], isSelected)[:num_backbone_atoms]
		backbone_y = torch.masked_select(coords_dst[0,:,1], isSelected)[:num_backbone_atoms]
		backbone_z = torch.masked_select(coords_dst[0,:,2], isSelected)[:num_backbone_atoms]
		backbone_coords = torch.stack([backbone_x, backbone_y, backbone_z], dim=1).resize_(1, num_backbone_atoms*3).contiguous()

		print(num_backbone_atoms)
		return 	backbone_coords.to(dtype=self.dtype, device=self.device), torch.tensor([num_backbone_atoms], dtype=torch.int, device=self.device)

	def setUp(self):
		print(self.msg)
		self.a2b = Angles2Backbone()
		self.rmsd = Coords2RMSD()
		self.ref_coords, self.num_atoms = self.prepare_structure()

		self.center = Coords2Center()
		self.translate = CoordsTranslate()
		self.rotate = CoordsRotate()

	def runTest(self):
		num_aa = torch.zeros(1, dtype=torch.int, device=self.device).fill_( int(self.num_atoms.item()/3) )
		angles = torch.zeros(1, 3, num_aa.item(), dtype=self.dtype, device=self.device).normal_().requires_grad_()
		param = self.a2b.get_default_parameters().to(device=self.device, dtype=self.dtype).requires_grad_()
					
		optimizer = optim.Adam([angles], lr = 0.05)

		loss_data = []
		g_src = []
		g_dst = []
	
		#Rotating for visualization convenience
		with torch.no_grad():
			backbone_coords = self.rotate(	self.ref_coords, 
											torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]], dtype=self.dtype, device=self.device), 
											self.num_atoms)
		
		for epoch in range(300):
			print('step', epoch)
			optimizer.zero_grad()
			coords_src = self.a2b(angles, param, num_aa)
			L = self.rmsd(coords_src, backbone_coords, self.num_atoms)
			L.backward()
			optimizer.step()
			

			loss_data.append(L.item())

			#Obtaining aligned structures
			with torch.no_grad():
				center_src = self.center(coords_src, self.num_atoms)
				center_dst = self.center(backbone_coords, self.num_atoms)
				c_src = self.translate(coords_src, -center_src, self.num_atoms)
				c_dst = self.translate(backbone_coords, -center_dst, self.num_atoms)
				rc_src = self.rotate(c_src, self.rmsd.UT.transpose(1,2).contiguous(), self.num_atoms)

				rc_src = rc_src.view( int(rc_src.size(1)/3), 3).cpu().numpy()
				c_dst = c_dst.view( int(c_dst.size(1)/3), 3).cpu().numpy()
			
				g_src.append(rc_src)
				g_dst.append(c_dst)

		fig = plt.figure()
		plt.title("Backbone fitting")
		plt.plot(loss_data)
		plt.xlabel("iteration")
		plt.ylabel("rmsd")
		plt.savefig("ExampleFitBackboneTraceParam.png")
		plt.show()

		#Plotting result
		fig = plt.figure()
		plt.title("Reduced model")
		ax = p3.Axes3D(fig)
		sx, sy, sz = g_src[0][:,0], g_src[0][:,1], g_src[0][:,2]
		rx, ry, rz = g_dst[0][:,0], g_dst[0][:,1], g_dst[0][:,2]
		line_src, = ax.plot(sx, sy, sz, 'b-', label = 'pred')
		line_dst, = ax.plot(rx, ry, rz, 'r.-', label = 'target')
			
		def update_plot(i):
			sx, sy, sz = g_src[i][:,0], g_src[i][:,1], g_src[i][:,2]
			rx, ry, rz = g_dst[i][:,0], g_dst[i][:,1], g_dst[i][:,2]
			line_src.set_data(sx, sy)
			line_src.set_3d_properties(sz)
			
			line_dst.set_data(rx, ry)
			line_dst.set_3d_properties(rz)
			
			return line_src, line_dst

		anim = animation.FuncAnimation(fig, update_plot,
                                    frames=300, interval=20, blit=True)
		ax.legend()
		anim.save("ExampleFitBackboneResultParam.gif", dpi=80, writer='imagemagick')
		plt.show()


		
if __name__=='__main__':
	unittest.main()