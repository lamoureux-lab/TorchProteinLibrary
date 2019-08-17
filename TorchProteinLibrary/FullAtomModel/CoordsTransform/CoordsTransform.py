import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module

import _FullAtomModel
import math

import sys
import os

def getBBox(input_coords, num_atoms):
	if len(input_coords.size())==2:
		batch_size = input_coords.size(0)
		a = torch.zeros(batch_size, 3, dtype=input_coords.dtype)
		b = torch.zeros(batch_size, 3, dtype=input_coords.dtype)
	else:
		raise ValueError('getBBox: ', 'Incorrect input size:', input_coords.size()) 

	_FullAtomModel.getBBox(input_coords, a, b, num_atoms)

	return a, b

def getRandomTranslation(a, b, volume_size):
	if len(a.size())==2:
		batch_size = a.size(0)
		T = torch.zeros(batch_size, 3, dtype=a.dtype)
	else:
		raise ValueError('getRandomTranslation: ', 'Incorrect input size:', a.size()) 

	_FullAtomModel.getRandomTranslation(T, a, b, volume_size)

	return T

def getRandomRotation(batch_size):
	R = torch.zeros(batch_size, 3, 3, dtype=torch.double)
	_FullAtomModel.getRandomRotation(R)
	return R

def getRotation(u):
	batch_size = u.size(0)
	R = torch.zeros(batch_size, 3, 3, dtype=torch.double)
	_FullAtomModel.getRotation(R, u)
	return R

class CoordsTranslateFunction(Function):
	"""
	coordinates translation
	"""
	@staticmethod	
	def forward(ctx, input_coords, T, num_atoms):
		ctx.save_for_backward(T, num_atoms)

		if len(input_coords.size())==2:
			batch_size = input_coords.size(0)
			num_coords = input_coords.size(1)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', input_coords.size()) 
		
		if input_coords.is_cuda:
			output_coords = torch.zeros(batch_size, num_coords, dtype=input_coords.dtype, device='cuda')
			_FullAtomModel.CoordsTranslateGPU_forward( input_coords, output_coords, T, num_atoms)	
		else:
			output_coords = torch.zeros(batch_size, num_coords, dtype=input_coords.dtype, device='cpu')
			_FullAtomModel.CoordsTranslate_forward( input_coords, output_coords, T, num_atoms)

		if math.isnan(output_coords.sum()):
			raise(Exception('CoordsTranslateFunction: forward Nan'))	

		return output_coords

	@staticmethod
	def backward(ctx, grad_output_coords):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords = grad_output_coords.contiguous()
		T, num_atoms = ctx.saved_tensors

		if len(grad_output_coords.size()) == 2:
			batch_size = grad_output_coords.size(0)
			num_coords = grad_output_coords.size(1)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', grad_output_coords.size()) 
		
		if grad_output_coords.is_cuda:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_coords.dtype, device='cuda')
			_FullAtomModel.CoordsTranslateGPU_backward(grad_output_coords, grad_input_coords, T, num_atoms)
		else:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_coords.dtype, device='cpu')
			_FullAtomModel.CoordsTranslate_backward(grad_output_coords, grad_input_coords, T, num_atoms)
						
		return grad_input_coords, None, None

class CoordsTranslate(Module):
	def __init__(self):
		super(CoordsTranslate, self).__init__()
		
	def forward(self, input_coords, T, num_atoms):
		return CoordsTranslateFunction.apply(input_coords, T, num_atoms)


class CoordsRotateFunction(Function):
	"""
	coordinates rotation
	"""
	@staticmethod					
	def forward(ctx, input_coords, R, num_atoms):
		ctx.save_for_backward(R, num_atoms)
		
		if len(input_coords.size())==2:
			batch_size = input_coords.size(0)
			num_coords = input_coords.size(1)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', input_coords.size()) 
		
		if input_coords.is_cuda:
			output_coords = torch.zeros(batch_size, num_coords, dtype=input_coords.dtype, device='cuda')
			_FullAtomModel.CoordsRotateGPU_forward( input_coords, output_coords, R, num_atoms)
		else:
			output_coords = torch.zeros(batch_size, num_coords, dtype=input_coords.dtype, device='cpu')
			_FullAtomModel.CoordsRotate_forward( input_coords, output_coords, R, num_atoms)

		if math.isnan(output_coords.sum()):
			raise(Exception('CoordsRotateFunction: forward Nan'))	

		return output_coords
	
	@staticmethod		
	def backward(ctx, grad_output_coords):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords = grad_output_coords.contiguous()
		R, num_atoms = ctx.saved_tensors

		if len(grad_output_coords.size()) == 2:
			batch_size = grad_output_coords.size(0)
			num_coords = grad_output_coords.size(1)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', grad_output_coords.size()) 
		
		if grad_output_coords.is_cuda:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_coords.dtype, device='cuda')
			_FullAtomModel.CoordsRotateGPU_backward(grad_output_coords, grad_input_coords, R, num_atoms)
		else:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_coords.dtype, device='cpu')
			_FullAtomModel.CoordsRotate_backward(grad_output_coords, grad_input_coords, R, num_atoms)
		
		if math.isnan(grad_input_coords.sum()):
			raise(Exception('CoordsRotateFunction: backward Nan'))		
		
		return grad_input_coords, None, None

class CoordsRotate(Module):
	def __init__(self):
		super(CoordsRotate, self).__init__()
		
	def forward(self, input_coords, R, num_atoms):
		return CoordsRotateFunction.apply(input_coords, R, num_atoms)

class Coords2CenterFunction(Function):
	"""
	Get coordinates geometrical center
	"""
	@staticmethod					
	def forward(ctx, input_coords, num_atoms):
		ctx.save_for_backward(input_coords, num_atoms)

		if len(input_coords.size())==2:
			batch_size = input_coords.size(0)
			num_coords = input_coords.size(1)
		else:
			raise ValueError('Coords2CenterFunction: ', 'Incorrect input size:', input_coords.size()) 
		
		if input_coords.is_cuda:
			output_center = torch.zeros(batch_size, 3, dtype=input_coords.dtype, device='cuda')
			_FullAtomModel.Coords2CenterGPU_forward( input_coords, output_center, num_atoms)
		else:
			output_center = torch.zeros(batch_size, 3, dtype=input_coords.dtype, device='cpu')
			_FullAtomModel.Coords2Center_forward( input_coords, output_center, num_atoms)

		if math.isnan(output_center.sum()):
			raise(Exception('Coords2CenterFunction: forward Nan'))	

		return output_center
	
	@staticmethod		
	def backward(ctx, grad_output_center):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_center = grad_output_center.contiguous()
		input_coords, num_atoms = ctx.saved_tensors

		if len(grad_output_center.size()) == 2:
			batch_size = grad_output_center.size(0)
			num_coords = input_coords.size(1)
		else:
			raise ValueError('Coords2CenterFunction: ', 'Incorrect input size:', grad_output_center.size()) 
		
		if grad_output_center.is_cuda:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_center.dtype, device='cuda')
			_FullAtomModel.Coords2CenterGPU_backward(grad_output_center, grad_input_coords, num_atoms)
		else:
			grad_input_coords = torch.zeros(batch_size, num_coords, dtype=grad_output_center.dtype, device='cpu')
			_FullAtomModel.Coords2Center_backward(grad_output_center, grad_input_coords, num_atoms)
		
		if math.isnan(grad_input_coords.sum()):
			raise(Exception('Coords2CenterFunction: backward Nan'))		
		
		return grad_input_coords, None

class Coords2Center(Module):
	def __init__(self):
		super(Coords2Center, self).__init__()
		
	def forward(self, input_coords, num_atoms):
		return Coords2CenterFunction.apply(input_coords, num_atoms)




