import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.module import Module
import math

import _Volume

import sys
import os

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def circ_volume_norm(circ_volume):
		batch_size = circ_volume.size(0)
		volume_size = circ_volume.size(1)
		L = volume_size
		L2 = int(volume_size/2)
		
		output_volume = torch.zeros(batch_size, volume_size, volume_size, volume_size, dtype=torch.float, device='cuda')
		# quadrants 1 <--> 8
		output_volume[:, :L2, :L2, :L2] = circ_volume[:, L2:L, L2:L, L2:L]
		output_volume[:, L2:L, L2:L, L2:L] = circ_volume[:, :L2, :L2, :L2]

		# quadrants 2 <--> 9
		output_volume[:, L2:L, :L2, :L2] = circ_volume[:, :L2, L2:L, L2:L]
		output_volume[:, :L2, L2:L, L2:L] = circ_volume[:, L2:L, :L2, :L2]

		# quadrants 3 <--> 5
		output_volume[:, L2:L, L2:L, :L2] = circ_volume[:, :L2, :L2, L2:L]
		output_volume[:, :L2, :L2, L2:L] = circ_volume[:, L2:L, L2:L, :L2]

		# quadrants 4 <--> 6
		output_volume[:, :L2, L2:L, :L2] = circ_volume[:, L2:L, :L2, L2:L]
		output_volume[:, L2:L, :L2, L2:L] = circ_volume[:, :L2, L2:L, :L2]

		return output_volume


class VolumeConvolutionFunction(Function):
	"""
	coordinates rotation
	"""
	@staticmethod					
	def forward(ctx, volume1, volume2, clip):
		ctx.save_for_backward(volume1, volume2)
		ctx.clip = clip

		if len(volume1.size())==4:
			batch_size = volume1.size(0)
			volume_size = volume1.size(1)
			output = torch.zeros(batch_size, volume_size, volume_size, volume_size, dtype=torch.float, device='cuda')
		else:
			raise ValueError('VolumeConvolutionFunction: ', 'Incorrect input size:', volume1.size()) 

		_Volume.VolumeConvolution_forward( volume1, volume2, output)
		
		if math.isnan(output.sum()):
			raise(Exception('VolumeConvolutionFunction: forward Nan'))

		return output
	
	@staticmethod		
	def backward(ctx, grad_output):
		# ATTENTION! It passes non-contiguous tensor
		grad_output = grad_output.contiguous()
		volume1, volume2 = ctx.saved_tensors

		if len(grad_output.size()) == 4:
			batch_size = grad_output.size(0)
			volume_size = grad_output.size(1)
			grad_input_volume1 = torch.zeros(batch_size, volume_size, volume_size, volume_size, dtype=torch.float, device='cuda')
			grad_input_volume2 = torch.zeros(batch_size, volume_size, volume_size, volume_size, dtype=torch.float, device='cuda')
		else:
			raise ValueError('VolumeConvolutionFunction: ', 'Incorrect input size:', grad_output.size()) 
	
		_Volume.VolumeConvolution_backward(grad_output, grad_input_volume1, grad_input_volume2, volume1, volume2)

		if not ctx.clip is None:
			grad_input_volume1 = grad_input_volume1.clamp(min=-math.fabs(ctx.clip), max=math.fabs(ctx.clip))
			grad_input_volume2 = grad_input_volume2.clamp(min=-math.fabs(ctx.clip), max=math.fabs(ctx.clip))
				
		if math.isnan(grad_input_volume1.sum()):
			raise(Exception('VolumeConvolutionFunction: backward Nan'))		
		
		return grad_input_volume1, grad_input_volume2, None

class VolumeConvolutionF(Module):
	def __init__(self, clip=None):
		super(VolumeConvolutionF, self).__init__()
		self.clip = clip
	def forward(self, input_volume1, input_volume2):
		return VolumeConvolutionFunction.apply(input_volume1, input_volume2, self.clip)

class VolumeConvolution(Module):
	def __init__(self, clip=None):
		super(VolumeConvolution, self).__init__()
		# self.W = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1, 1, dtype=torch.float, device='cuda').normal_())
		self.convolve = VolumeConvolutionF(clip)
		
	def forward(self, input_volume1, input_volume2):
		batch_size = input_volume1.size(0)
		num_features = input_volume1.size(1)
		volume_size = input_volume1.size(2)
		# print(input_volume1.size())
		input_volume1 = F.pad(input_volume1, (0, volume_size, 0, volume_size, 0, volume_size)).contiguous()
		input_volume2 = F.pad(input_volume2, (0, volume_size, 0, volume_size, 0, volume_size)).contiguous()
		# print(input_volume1.size())
		volume1 = input_volume1.reshape(batch_size*num_features, 2*volume_size, 2*volume_size, 2*volume_size).contiguous()
		volume2 = input_volume2.reshape(batch_size*num_features, 2*volume_size, 2*volume_size, 2*volume_size).contiguous()
		# print(volume1.size())
		output = self.convolve(volume1, volume2)
		
		output = output.reshape(batch_size, num_features, 2*volume_size, 2*volume_size, 2*volume_size).contiguous()
		# output = output*self.W
		
		return output