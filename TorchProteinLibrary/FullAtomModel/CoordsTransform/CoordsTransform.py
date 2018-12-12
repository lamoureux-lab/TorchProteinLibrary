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
		a = torch.zeros(batch_size, 3, dtype=torch.double)
		b = torch.zeros(batch_size, 3, dtype=torch.double)
	else:
		raise ValueError('getBBox: ', 'Incorrect input size:', input_coords.size()) 

	_FullAtomModel.getBBox(input_coords, a, b, num_atoms)

	return a, b

def getRandomTranslation(a, b, volume_size):
	if len(a.size())==2:
		batch_size = a.size(0)
		T = torch.zeros(batch_size, 3, dtype=torch.double)
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
	def forward(ctx, input_coords_cpu, T, num_atoms):
		# ctx.save_for_backward(num_atoms)

		if len(input_coords_cpu.size())==2:
			batch_size = input_coords_cpu.size(0)
			num_coords = input_coords_cpu.size(1)
			output_coords_cpu = torch.zeros(batch_size, num_coords, dtype=torch.double)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		_FullAtomModel.CoordsTranslate_forward( input_coords_cpu, output_coords_cpu, T, num_atoms)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('CoordsTranslateFunction: forward Nan'))	

		return output_coords_cpu

	@staticmethod
	def backward(ctx, grad_output_coords_cpu):
		# ATTENTION! It passes non-contiguous tensor
		grad_output_coords_cpu = grad_output_coords_cpu.contiguous()

		if len(grad_output_coords_cpu.size()) == 2:
			batch_size = grad_output_coords_cpu.size(0)
			num_coords = grad_output_coords_cpu.size(1)
			grad_input_coords_cpu = torch.zeros(batch_size, num_coords, dtype=torch.double)
		else:
			raise ValueError('CoordsTranslateFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		grad_input_coords_cpu.data.copy_(grad_output_coords_cpu)
				
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
			output_coords_cpu = torch.zeros(batch_size, num_coords, dtype=torch.double)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', input_coords_cpu.size()) 

		_FullAtomModel.CoordsRotate_forward( input_coords_cpu, output_coords_cpu, R, num_atoms)

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
			grad_input_coords_cpu = torch.zeros(batch_size, num_coords, dtype=torch.double)
		else:
			raise ValueError('CoordsRotateFunction: ', 'Incorrect input size:', input_angles_cpu.size()) 
		
		_FullAtomModel.CoordsRotate_backward(grad_output_coords_cpu, grad_input_coords_cpu, R, num_atoms)
		
		if math.isnan(grad_input_coords_cpu.sum()):
			raise(Exception('CoordsRotateFunction: backward Nan'))		
		
		return grad_input_coords_cpu, None, None

class CoordsRotate(Module):
	def __init__(self):
		super(CoordsRotate, self).__init__()
		
	def forward(self, input_coords_cpu, R, num_atoms):
		return CoordsRotateFunction.apply(input_coords_cpu, R, num_atoms)

class Coords2CenteredCoords(Module):
	def __init__(self, rotate=True, translate=True, box_size=120, resolution=1.0):
		super(Coords2CenteredCoords, self).__init__()
		self.rotate = rotate
		self.translate = translate
		self.box_size = box_size
		self.box_length = box_size * resolution
		self.resolution = resolution
		self.rot = CoordsRotate()
		self.tra = CoordsTranslate()

	def forward(self, input_coords, num_atoms):
		batch_size = input_coords.size(0)
		a,b = getBBox(input_coords, num_atoms)
		protein_center = (a+b)*0.5
		coords = self.tra(input_coords, -protein_center, num_atoms)

		box_center = torch.zeros(batch_size, 3, dtype=torch.double)
		box_center.fill_(self.box_length/2.0)
		
		if self.rotate:	
			rR = getRandomRotation(batch_size)
			coords = self.rot(coords, rR, num_atoms)
		
		coords = self.tra(coords, box_center, num_atoms)

		if self.translate:
			a,b = getBBox(coords, num_atoms)
			rT = getRandomTranslation(a, b, self.box_length)
			coords = self.tra(coords, rT, num_atoms)

		return coords



