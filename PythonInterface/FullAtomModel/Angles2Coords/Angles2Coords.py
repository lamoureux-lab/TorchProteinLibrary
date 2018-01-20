import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2Coords
import math

class Angles2CoordsFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	def __init__(self, sequence):
		super(Angles2CoordsFunction, self).__init__()
		self.sequence = sequence
				
	def forward(self, input_angles_cpu):
		
		if len(input_angles_cpu.size())==2:
			output_coords_cpu = torch.DoubleTensor(3*18*len(self.sequence))
			
		elif len(input_angles_cpu.size())==3:
			raise(Exception('Not implemented'))

		else:
			raise ValueError('Angles2CoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 

		output_coords_cpu.fill_(0.0)
		cppAngles2Coords.Angles2Coords_forward( self.sequence,
												input_angles_cpu,              #input angles
												output_coords_cpu,  #output coordinates
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
		cppAngles2Coords.Angles2Coords_backward(grad_atoms_cpu, grad_angles_cpu, self.sequence, input_angles_cpu)
		
		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		
		
		return grad_angles_cpu

class Angles2Coords(Module):
	def __init__(self, sequence):
		super(Angles2Coords, self).__init__()
		self.sequence = sequence
		
	def forward(self, input_angles_cpu):
		return Angles2CoordsFunction(self.sequence)(input_angles_cpu)