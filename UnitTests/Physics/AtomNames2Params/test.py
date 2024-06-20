import os
import sys
import torch
import numpy as np
import unittest
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.Physics import AtomNames2Params, ElectrostaticParameters

class TestAtomNames2Params(unittest.TestCase):
	device = 'cpu'
	dtype = torch.double
	places = 7
	batch_size = 16
	max_num_atoms = 30
	eps=1e-06 
	atol=1e-05 
	rtol=0.001
	msg = "Testing electrostatics"

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

		param_charge_text="""N     ALA      -0.4157
H     ALA       0.2719
CA    ALA       0.0337
HA    ALA       0.0823
CB    ALA      -0.1825
1HB   ALA       0.0603
2HB   ALA       0.0603
3HB   ALA       0.0603
HB1   ALA       0.0603
HB2   ALA       0.0603
HB3   ALA       0.0603
C     ALA       0.5973
O     ALA      -0.5679

N     GLN      -0.4157
H     GLN       0.2719
CA    GLN      -0.0031
HA    GLN       0.0850
CB    GLN      -0.0036
1HB   GLN       0.0171
HB1   GLN       0.0171
2HB   GLN       0.0171
HB2   GLN       0.0171 
3HB   GLN       0.0171
HB3   GLN       0.0171
CG    GLN      -0.0645
1HG   GLN       0.0352
HG1   GLN       0.0352
2HG   GLN       0.0352
HG2   GLN       0.0352 
3HG   GLN       0.0352
HG3   GLN       0.0352
CD    GLN       0.6951
1OE   GLN      -0.6086
OE1   GLN      -0.6086
2NE   GLN      -0.9407
NE2   GLN      -0.9407
1HE2  GLN       0.4251
2HE2  GLN       0.4251
HE21  GLN       0.4251 
HE22  GLN       0.4251 
C     GLN       0.5973
O     GLN      -0.5679

N     THR      -0.4157
H     THR       0.2719
CA    THR      -0.0389
HA    THR       0.1007
CB    THR       0.3654
HB    THR       0.0043
OG1   THR      -0.6761
1OG   THR      -0.6761
HG1   THR       0.4102
1HG   THR       0.4102
CG2   THR      -0.2438
1HG2  THR       0.0642
2HG2  THR       0.0642
3HG2  THR       0.0642
HG21  THR       0.0642 
HG22  THR       0.0642 
HG23  THR       0.0642 
C     THR       0.5973
O     THR      -0.5679
"""
		with open("amber.crg", "w") as fout:
			fout.write(param_charge_text)

		param_size_text = """N     ALA   1.8240
H     ALA   0.6000
CA    ALA   1.9080
HA    ALA   1.3870
CB    ALA   1.9080
1HB   ALA   1.4870
2HB   ALA   1.4870
3HB   ALA   1.4870
HB1   ALA   1.4870
HB2   ALA   1.4870
HB3   ALA   1.4870
C     ALA   1.9080
O     ALA   1.6612

N     GLN   1.8240
H     GLN   0.6000
CA    GLN   1.9080
HA    GLN   1.3870
CB    GLN   1.9080
1HB   GLN   1.4870
2HB   GLN   1.4870
3HB   GLN   1.4870
HB1   GLN   1.4870
HB2   GLN   1.4870
HB3   GLN   1.4870
CG    GLN   1.9080
1HG   GLN   1.4870
2HG   GLN   1.4870
3HG   GLN   1.4870
HG1   GLN   1.4870
HG2   GLN   1.4870
HG3   GLN   1.4870
CD    GLN   1.9080
OE1   GLN   1.6612
NE2   GLN   1.8240
1HE2  GLN   0.6000
2HE2  GLN   0.6000
HE21  GLN   0.6000
HE22  GLN   0.6000
C     GLN   1.9080
O     GLN   1.6612
H1    GLN   0.6000
H2    GLN   0.6000
H3    GLN   0.6000

N     THR   1.8240
H     THR   0.6000
CA    THR   1.9080
HA    THR   1.3870
CB    THR   1.9080
HB    THR   1.3870
OG1   THR   1.7210
HG1   THR   0.6000 !HG1   THR   0.0001
CG2   THR   1.9080
1OG   THR   1.7210
1HG   THR   0.6000 !1HG   THR   0.0001
2CG   THR   1.9080
1HG2  THR   1.4870
2HG2  THR   1.4870
3HG2  THR   1.4870
HG21  THR   1.4870
HG22  THR   1.4870
HG23  THR   1.4870
C     THR   1.9080
O     THR   1.6612
"""
		with open("amber.siz", "w") as fout:
			fout.write(param_size_text)
		self.elec_params = ElectrostaticParameters('.', type='amber')
		self.a2p = AtomNames2Params()


class TestAtomNames2Params_forward(TestAtomNames2Params):

	def runTest(self):
		coords, chain_names, resnames, resnums, anames, num_atoms = self.p2c(["test.pdb"])
		params = self.a2p(resnames, anames, num_atoms, self.elec_params.types, self.elec_params.params)
		
		index = 0
		with open("test.pdb", "r") as fin:
			for line in fin:
				sline = line.split()
				atomname = sline[2]
				resname = sline[3]
				charge, radius = self.elec_params.param_dict[(resname, atomname)]
				test_charge = params[0, index, 0]
				test_radius = params[0, index, 1]
				self.assertEqual(charge, test_charge.item())
				self.assertEqual(radius, test_radius.item())
				index += 1

class TestAtomNames2Params_backward(TestAtomNames2Params):

	def runTest(self):
		coords, chain_names, resnames, resnums, anames, num_atoms = self.p2c(["test.pdb"])
		types = self.elec_params.types
		params = self.elec_params.params.requires_grad_()
		result = torch.autograd.gradcheck(self.a2p, (resnames, anames, num_atoms, types, params), self.eps, self.atol, self.rtol)
		self.assertTrue(result)

if __name__ == '__main__':
	unittest.main()