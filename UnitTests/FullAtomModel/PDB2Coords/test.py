import os
import sys
import torch
from TorchProteinLibrary import FullAtomModel
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered

class TestPDB2Coords(unittest.TestCase):
	def _convert2str(self, tensor):
		return tensor.numpy().astype(dtype=np.uint8).tostring().split(b'\00')[0]

	def _plot_coords(self, coords, filename):
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
		self.p2c = PDB2CoordsUnordered()
		pdb_text = """ATOM      0    N GLN A   0       0.000   0.000   0.000
ATOM      1   CA GLN A   0       0.526   0.000  -1.362
ATOM      2   CB GLN A   0       1.349  -1.258  -1.588
ATOM      3   CG GLN A   0       1.488  -1.511  -3.080
ATOM      4   CD GLN A   0       2.281  -0.380  -3.716
ATOM      5  OE1 GLN A   0       3.338  -0.604  -4.304
ATOM      6  NE2 GLN A   0       1.770   0.842  -3.597
ATOM      7    C GLN A   0      -0.471   0.000  -2.516
ATOM      8    O GLN A   0      -1.468  -1.003  -2.607
ATOM      9    N THR A   1      -0.434   0.934  -3.463
ATOM     10   CA THR A   1      -1.494   0.728  -4.446
ATOM     11   CB THR A   1      -2.120   2.065  -4.806
ATOM     12  OG1 THR A   1      -1.960   2.961  -3.719
ATOM     13  CG2 THR A   1      -1.286   2.785  -5.863
ATOM     14    C THR A   1      -1.115   0.124  -5.794
ATOM     15    O THR A   1      -0.118   0.730  -6.599
ATOM     16    N ALA A   2      -1.703  -0.979  -6.250
ATOM     17   CA ALA A   2      -1.162  -1.365  -7.550
ATOM     18   CB ALA A   2      -1.839  -2.641  -8.022
ATOM     19    C ALA A   2      -1.340  -0.399  -8.717
ATOM     20    O ALA A   2      -2.433   0.502  -8.744
ATOM     21    N ALA A   3      -0.482  -0.375  -9.733
ATOM     22   CA ALA A   3      -0.858   0.638 -10.714
ATOM     23   CB ALA A   3      -0.125   0.377 -12.020
ATOM     24    C ALA A   3      -2.324   0.730 -11.127
ATOM     25    O ALA A   3      -3.026   1.956 -11.020"		
"""
		with open("test.pdb", "w") as fout:
			fout.write(pdb_text)
		
class TestCoords2PDBUnordered(TestPDB2Coords):
	def runTest(self):
		coords, chain_names, resnames, resnums, anames, num_atoms = self.p2c(["test.pdb"])
		
		#num_atoms
		self.assertEqual(num_atoms[0].item(), 26)
		#chain names
		for i in range(num_atoms[0].item()):
			self.assertEqual(self._convert2str(chain_names[0,i,:]), b'A')
		#residue names
		for i in range(9):
			self.assertEqual(self._convert2str(resnames[0,i,:]), b'GLN')
		for i in range(9,16):
			self.assertEqual(self._convert2str(resnames[0,i,:]), b'THR')
		for i in range(16,25):
			self.assertEqual(self._convert2str(resnames[0,i,:]), b'ALA')
		
		#atom names
		ref_atom_set = set([b'N', b'C', b'O', b'CA', b'CB', b'CD', b'CG', b'OG1', b'CG2', b'OE1', b'NE2'])
		sam_atom_set = set([])
		for i in range(num_atoms[0].item()):
			sam_atom_set.add(self._convert2str(anames[0,i,:]))
		
		self.assertTrue((ref_atom_set | sam_atom_set) == ref_atom_set )
		self.assertTrue((ref_atom_set & sam_atom_set) == ref_atom_set )

		#coords
		coords = coords.reshape(1, num_atoms[0].item(), 3)
		self.assertAlmostEqual(coords[0,10,0].item(), -1.494, places = 4)
		self.assertAlmostEqual(coords[0,10,1].item(), 0.728, places = 4)
		self.assertAlmostEqual(coords[0,10,2].item(), -4.446, places = 4)
		


if __name__=='__main__':
	unittest.main()
