import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
from .utils import transform, bytes2string
import numpy as np
from .rotamers import getAngles, generateAA
from Bio.PDB.Polypeptide import aa1
import unittest


class TestAngles2CoordsForward(unittest.TestCase):

	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()

	def _plot_aa(self, filename):
		import matplotlib as mpl
		mpl.use('Agg')
		import seaborn as sea
		import mpl_toolkits.mplot3d.axes3d as p3
		from matplotlib import pylab as plt
		
		if not os.path.exists("TestFig"):
			os.mkdir("TestFig")

		sx, sy, sz = [], [], []
		rx, ry, rz = [], [], []
		for atom_name in self.reference.keys():
			rx.append(self.reference[atom_name][0])
			ry.append(self.reference[atom_name][1])
			rz.append(self.reference[atom_name][2])
			sx.append(self.sample[atom_name][0])
			sy.append(self.sample[atom_name][1])
			sz.append(self.sample[atom_name][2])
			
		all_xyz = rx + ry + rz
		min_xyz = np.min(all_xyz)
		max_xyz = np.max(all_xyz)
		fig = plt.figure()
		ax = p3.Axes3D(fig)
		ax.plot(sx, sy, sz, '.', label = "sample")
		ax.plot(rx, ry, rz, 'x', label = "reference")
		ax.set_xlim(min_xyz,max_xyz)
		ax.set_ylim(min_xyz,max_xyz)
		ax.set_zlim(min_xyz,max_xyz)
		ax.legend()
		plt.savefig('TestFig/%s'%filename)

	def _generate_reference(self, aa):
		self.reference = {}
		structure = generateAA(aa)
		structure = transform(structure)
		self.angles = getAngles(structure)
		for atom in structure.get_atoms():
			self.reference[atom.get_name()] = np.array(atom.get_coord())
	
	def _generate_sample(self, aa):
		sequences = [aa]
		protein, res_names, atom_names, num_atoms = self.a2c(self.angles, sequences)
		self.sample = {}
		for i in range(atom_names.size(1)):
			self.sample[bytes2string(atom_names[0,i,:])] = protein.data[0,3*i : 3*i + 3].numpy()

	def _measure_rmsd(self):
		N = len(self.reference.keys())
		RMSD = 0.0
		for atom_name in self.reference.keys():
			norm = np.linalg.norm(self.reference[atom_name] - self.sample[atom_name])
			RMSD += norm*norm
		RMSD /= float(N*(N-1))
		return np.sqrt(RMSD)

	def runTest(self):
		for aa in aa1:
			self._generate_reference(aa)
			self._generate_sample(aa)
			self._plot_aa(aa+"_forward.png")
			rmsd = self._measure_rmsd()
			self.assertLess(rmsd, 1.0)
	
class TestAngles2CoordsBackward(unittest.TestCase):

	def setUp(self):
		self.a2c = FullAtomModel.Angles2Coords()

	def _plot_curve(self, sample, reference, filename):
		import matplotlib as mpl
		mpl.use('Agg')
		import seaborn as sea
		import mpl_toolkits.mplot3d.axes3d as p3
		from matplotlib import pylab as plt
		
		if not os.path.exists("TestFig"):
			os.mkdir("TestFig")
		f = plt.figure()
		plt.plot(sample, "--r", label='sample')
		plt.plot(reference, ".-b", label='reference')
		plt.legend()
		plt.savefig('TestFig/%s'%filename)

	def runTest(self):
		sequences = ['ACDEFGHIKLMNPQRSTVWY']
		x0 = torch.zeros(len(sequences), 8, len(sequences[-1]), dtype=torch.double, device='cpu').requires_grad_()
		x1 = torch.zeros(len(sequences), 8, len(sequences[-1]), dtype=torch.double, device='cpu')
		x0.data[:,0,:] = -1.047
		x0.data[:,1,:] = -0.698
		x0.data[:,2,:] = np.pi
		x0.data[:,3:,:] = 110.4*np.pi/180.0
		
		y0, res, at, n_at = self.a2c(x0, sequences)
		y0 = y0.sum()
			
		y0.backward()
		back_grad_x0 = torch.zeros(x0.grad.size()).copy_(x0.grad)
		
		error = 0.0
		N = 0
		for b in range(0,len(sequences)):
			for i in range(0,8):
				reference = back_grad_x0[b, i, :].numpy()
				sample = []
				for j in range(0,x0.size(2)):
					dx = 0.00001
					x1.copy_(x0)
					x1[b,i,j] += dx
					y1, res, at, n_at = self.a2c(x1,sequences)
					y1 = y1.sum()
					dy_dx = (y1.item()-y0.item())/(dx)
					sample.append(dy_dx)
					error += np.abs(dy_dx - back_grad_x0[b,i,j].item())
					N+=1
				self._plot_curve(sample, reference, "backward_%d.png"%(i))

		error/=float(N)
		#print('Error = ', error)
		self.assertLess(error, 0.01)
	
if __name__=='__main__':
	unittest.main()
	


