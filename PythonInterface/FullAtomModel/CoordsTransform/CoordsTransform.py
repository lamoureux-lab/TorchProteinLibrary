import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppCoordsTransform
import math

import sys
import os

def getBBox(input_coords, num_atoms):
	if len(input_coords.size())==2:
		batch_size = input_coords.size(0)
		a = torch.DoubleTensor(batch_size, 3)
		b = torch.DoubleTensor(batch_size, 3)
	else:
		raise ValueError('getBBox: ', 'Incorrect input size:', input_coords.size()) 

	cppCoordsTransform.getBBox(input_coords.data, a, b, num_atoms.data)

	return Variable(a), Variable(b)

def getRandomTranslation(a, b, volume_size):
	if len(a.size())==2:
		batch_size = a.size(0)
		T = torch.DoubleTensor(batch_size, 3)
	else:
		raise ValueError('getRandomTranslation: ', 'Incorrect input size:', a.size()) 

	cppCoordsTransform.getRandomTranslation(T, a.data, b.data, volume_size)

	return Variable(T)

def getRandomRotation(batch_size):
	R = torch.DoubleTensor(batch_size, 3, 3)
	cppCoordsTransform.getRandomRotation(R)
	return Variable(R)

class CoordsTranslateFunction(Function):
	"""
	coordinates translation
	"""
	@staticmethod	
	def forward(ctx, input_coords_cpu, T, num_atoms):
		# ctx.save_for_backward(num_atoms)

		if len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			num_coords = input_coords_cpu.size(1)
			output_coords_cpu = torch.DoubleTensor(batch_size, num_coords)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		cppCoordsTransform.CoordsTranslate_forward( input_coords_cpu, output_coords_cpu, T, num_atoms)

		# if math.isnan(output_coords_cpu.sum()):
		# 	raise(Exception('CoordsTranslateFunction: forward Nan'))	

		return output_coords_cpu

	@staticmethod
	def backward(ctx, grad_output_coords_cpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords_cpu = grad_output_coords_cpu.contiguous()

		if len(grad_output_coords_cpu.size()) == 2:
			batch_size = grad_output_coords_cpu.size(0)
			num_coords = grad_output_coords_cpu.size(1)
			grad_input_coords_cpu = torch.DoubleTensor(batch_size, num_coords)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		cppCoordsTransform.CoordsTranslate_backward(grad_output_coords_cpu, grad_input_coords_cpu)
		
		# if math.isnan(grad_input_coords_cpu.sum()):
		# 	raise(Exception('CoordsTranslateFunction: backward Nan'))		
		
		return grad_input_coords_cpu, None, None

class CoordsTranslate(Module):
	def __init__(self):
		super(CoordsTranslate, self).__init__()
		
	def forward(self, input_coords_cpu, T, num_atoms):
		return CoordsTranslateFunction.apply(input_coords_cpu, T, num_atoms)


class CoordsRotateFunction(Function):
	"""
	coordinates rotation
	"""
	@staticmethod					
	def forward(ctx, input_coords_cpu, R, num_atoms):
		ctx.save_for_backward(R, num_atoms)

		if len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			num_coords = input_coords_cpu.size(1)
			output_coords_cpu = torch.DoubleTensor(batch_size, num_coords)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		cppCoordsTransform.CoordsRotate_forward( input_coords_cpu, output_coords_cpu, R, num_atoms)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('CoordsRotateFunction: forward Nan'))	

		return output_coords_cpu
	
	@staticmethod		
	def backward(ctx, grad_output_coords_cpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords_cpu = grad_output_coords_cpu.contiguous()
		R, num_atoms = ctx.saved_tensors

		if len(grad_output_coords_cpu.size()) == 2:
			batch_size = grad_output_coords_cpu.size(0)
			num_coords = grad_output_coords_cpu.size(1)
			grad_input_coords_cpu = torch.DoubleTensor(batch_size, num_coords)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		cppCoordsTransform.CoordsRotate_backward(grad_output_coords_cpu, grad_input_coords_cpu, R, num_atoms)
		
		if math.isnan(grad_input_coords_cpu.sum()):
			raise(Exception('CoordsRotateFunction: backward Nan'))		
		
		return grad_input_coords_cpu, None, None

class CoordsRotate(Module):
	def __init__(self):
		super(CoordsRotate, self).__init__()
		
	def forward(self, input_coords_cpu, R, num_atoms):
		return CoordsRotateFunction.apply(input_coords_cpu, R, num_atoms)