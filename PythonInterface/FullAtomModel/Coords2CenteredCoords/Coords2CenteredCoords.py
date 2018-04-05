import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppCoords2CenteredCoords
import math

import sys
import os

class Coords2CenteredCoordsFunction(Function):
	"""
	coordinates -> coordinated arranged in atom types function
	"""
	def __init__(self, volume_size, rotate=False, translate=False):
		super(Coords2CenteredCoordsFunction, self).__init__()
		self.volume_size = volume_size
		self.rotate = rotate
		self.translate = translate
						
	def forward(self, input_coords_cpu):
		
		if len(input_coords_cpu.size())==1:
			num_coords = input_coords_cpu.size(0)
			output_coords_cpu = torch.DoubleTensor(num_coords)
			self.R = torch.DoubleTensor(3,3)
			self.T = torch.DoubleTensor(3)
			
		elif len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			num_coords = input_coords_cpu.size(1)
			output_coords_cpu = torch.DoubleTensor(batch_size, num_coords)
			self.R = torch.DoubleTensor(batch_size, 3,3)
			self.T = torch.DoubleTensor(batch_size, 3)

		else:
			raise ValueError('Coords2CenteredCoordsFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		
		cppCoords2CenteredCoords.Coords2CenteredCoords_forward( input_coords_cpu, output_coords_cpu, self.volume_size, self.R, self.T, self.rotate, self.translate)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Coords2CenteredCoordsFunction: forward Nan'))	

		return output_coords_cpu
			
	def backward(self, grad_output_coords_cpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords_cpu = grad_output_coords_cpu.contiguous()

		if len(grad_output_coords_cpu.size()) == 1:
			num_coords = grad_output_coords_cpu.size(0)
			grad_input_coords_cpu = torch.DoubleTensor(num_coords)
					
		elif len(grad_output_coords_cpu.size()) == 2:
			batch_size = grad_output_coords_cpu.size(0)
			num_coords = grad_output_coords_cpu.size(1)
			grad_input_coords_cpu = torch.DoubleTensor(batch_size, num_coords)

		else:
			raise ValueError('Coords2CenteredCoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		cppCoords2CenteredCoords.Coords2CenteredCoords_backward(grad_output_coords_cpu, grad_input_coords_cpu, self.R, self.T, self.rotate, self.translate)
		
		if math.isnan(grad_input_coords_cpu.sum()):
			print grad_output_coords_cpu.sum(), grad_input_coords_cpu.sum()
			raise(Exception('Coords2CenteredCoordsFunction: backward Nan'))		
		
		return grad_input_coords_cpu

class Coords2CenteredCoords(Module):
	def __init__(self, volume_size=120, rotate=False, translate=False):
		super(Coords2CenteredCoords, self).__init__()
		self.volume_size = volume_size
		self.rotate = rotate
		self.translate = translate
		
	def forward(self, input_coords_cpu):
		return Coords2CenteredCoordsFunction(self.volume_size, self.rotate, self.translate)(input_coords_cpu)