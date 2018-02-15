import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2Coords
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PDB2Coords import cppPDB2Coords

class Angles2CoordsFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	def __init__(self, sequence, num_atoms, add_term=False):
		super(Angles2CoordsFunction, self).__init__()
		self.sequence = sequence
		self.num_atoms = num_atoms
		self.add_term = add_term
		
				
	def forward(self, input_angles_cpu):
		
		if len(input_angles_cpu.size())==2:
			output_coords_cpu = torch.DoubleTensor(3*self.num_atoms)
			
		elif len(input_angles_cpu.size())==3:
			raise(Exception('Not implemented'))

		else:
			raise ValueError('Angles2CoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 

		output_coords_cpu.fill_(0.0)
		cppAngles2Coords.Angles2Coords_forward( self.sequence,
												input_angles_cpu,              #input angles
												output_coords_cpu,  #output coordinates
												self.add_term
												)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: forward Nan'))		
		
		self.save_for_backward(input_angles_cpu)

		return output_coords_cpu
			

	def backward(self, grad_atoms_cpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_atoms_cpu = grad_atoms_cpu.contiguous()
		
		input_angles_cpu, = self.saved_tensors
		if len(input_angles_cpu.size()) == 2:
			grad_angles_cpu = torch.DoubleTensor(input_angles_cpu.size(0), input_angles_cpu.size(1))
					
		elif len(input_angles_cpu.size()) == 3:
			raise(Exception('Not implemented'))

		else:
			raise ValueError('Angles2CoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		grad_angles_cpu.fill_(0.0)
		cppAngles2Coords.Angles2Coords_backward(grad_atoms_cpu, grad_angles_cpu, self.sequence, input_angles_cpu, self.add_term)
		
		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		
		
		return grad_angles_cpu

class Angles2Coords(Module):
	def __init__(self, sequence, num_atoms=None, add_term=False):
		super(Angles2Coords, self).__init__()
		self.sequence = sequence
		self.num_atoms = num_atoms
		self.add_term = add_term
		if self.num_atoms is None:
			self.num_atoms = cppPDB2Coords.getSeqNumAtoms(self.sequence, self.add_term)

		
	def forward(self, input_angles_cpu):
		return Angles2CoordsFunction(self.sequence, self.num_atoms, self.add_term)(input_angles_cpu)