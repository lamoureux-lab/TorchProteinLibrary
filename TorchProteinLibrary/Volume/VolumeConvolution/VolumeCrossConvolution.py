import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math

import _Volume

import sys
import os

def convolve(volume1, volume2, conj):
	batch_size = volume1.size(0)
	box_size = volume1.size(1)
	full_vol = box_size*box_size*box_size
	output = torch.zeros(batch_size, box_size, box_size, box_size, device=volume1.device, dtype=volume1.dtype)
	
	cv1 = torch.rfft(volume1, 3)
	cv2 = torch.rfft(volume2, 3)
	
	if conj:
		re = cv1[:,:,:,:,0]*cv2[:,:,:,:,0] + cv1[:,:,:,:,1]*cv2[:,:,:,:,1]
		im = -cv1[:,:,:,:,0]*cv2[:,:,:,:,1] + cv1[:,:,:,:,1]*cv2[:,:,:,:,0]
	else:
		re = cv1[:,:,:,:,0]*cv2[:,:,:,:,0] - cv1[:,:,:,:,1]*cv2[:,:,:,:,1]
		im = cv1[:,:,:,:,0]*cv2[:,:,:,:,1] + cv1[:,:,:,:,1]*cv2[:,:,:,:,0]
	
	cconv = torch.stack([re, im], dim=4)
	
	return torch.irfft(cconv, 3, signal_sizes=(box_size, box_size, box_size))

class CrossConvFunction(Function):
	@staticmethod
	def forward(ctx, volume1, volume2):
		ctx.save_for_backward(volume1, volume2)
		return convolve(volume1, volume2, conj=True)

	@staticmethod
	def backward(ctx, grad_output):
		volume1, volume2 = ctx.saved_tensors
		gradVolume1 = convolve(grad_output, volume2, conj=False)
		gradVolume2 = convolve(volume1, grad_output, conj=True)
		return gradVolume1, gradVolume2

class SwapQuadrants3DFunction(Function):
	@staticmethod
	def forward(ctx, input_volume):
		batch_size = input_volume.size(0)
		num_features = input_volume.size(1)
		L = input_volume.size(2)
		L2 = int(L/2)
		output_volume = torch.zeros(batch_size, num_features, L, L, device=input_volume.device, dtype=input_volume.dtype)
		
		# quadrants 1 <--> 8
		output_volume[:, :L2, :L2, :L2] = input_volume[:, L2:L, L2:L, L2:L]
		output_volume[:, L2:L, L2:L, L2:L] = input_volume[:, :L2, :L2, :L2]

		# quadrants 2 <--> 9
		output_volume[:, L2:L, :L2, :L2] = input_volume[:, :L2, L2:L, L2:L]
		output_volume[:, :L2, L2:L, L2:L] = input_volume[:, L2:L, :L2, :L2]

		# quadrants 3 <--> 5
		output_volume[:, L2:L, L2:L, :L2] = input_volume[:, :L2, :L2, L2:L]
		output_volume[:, :L2, :L2, L2:L] = input_volume[:, L2:L, L2:L, :L2]

		# quadrants 4 <--> 6
		output_volume[:, :L2, L2:L, :L2] = input_volume[:, L2:L, :L2, L2:L]
		output_volume[:, L2:L, :L2, L2:L] = input_volume[:, :L2, L2:L, :L2]

		return output_volume

	@staticmethod
	def backward(ctx, grad_output):
		batch_size = grad_output.size(0)
		num_features = grad_output.size(1)
		L = grad_output.size(2)
		L2 = int(L/2)
		grad_input = torch.zeros(batch_size, num_features, L, L, device=grad_output.device, dtype=grad_output.dtype)

		# quadrants 1 <--> 8
		grad_input[:, :L2, :L2, :L2] = grad_output[:, L2:L, L2:L, L2:L]
		grad_input[:, L2:L, L2:L, L2:L] = grad_output[:, :L2, :L2, :L2]

		# quadrants 2 <--> 9
		grad_input[:, L2:L, :L2, :L2] = grad_output[:, :L2, L2:L, L2:L]
		grad_input[:, :L2, L2:L, L2:L] = grad_output[:, L2:L, :L2, :L2]

		# quadrants 3 <--> 5
		grad_input[:, L2:L, L2:L, :L2] = grad_output[:, :L2, :L2, L2:L]
		grad_input[:, :L2, :L2, L2:L] = grad_output[:, L2:L, L2:L, :L2]

		# quadrants 4 <--> 6
		grad_input[:, :L2, L2:L, :L2] = grad_output[:, L2:L, :L2, L2:L]
		grad_input[:, L2:L, :L2, L2:L] = grad_output[:, :L2, L2:L, :L2]

		return grad_input


class VolumeCrossConvolution(nn.Module):
	def __init__(self, append_coords=False):
		super(VolumeCrossConvolution, self).__init__()
			
	def forward(self, volume1, volume2):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		
		# volume1_unpacked = []
		# volume2_unpacked = []
		# for i in range(0, num_features):
		# 	volume1_unpacked.append(volume1[:,0:num_features-i,:,:,:])
		# 	volume2_unpacked.append(volume2[:,i:num_features,:,:,:])
		# volume1 = torch.cat(volume1_unpacked, dim=1)
		# volume2 = torch.cat(volume2_unpacked, dim=1)

		num_output_features = num_features*num_features
		volume1 = volume1.unsqueeze(dim=1).repeat(1, num_features, 1, 1, 1, 1).contiguous()
		volume2 = volume2.unsqueeze(dim=2).repeat(1, 1, num_features, 1, 1, 1).contiguous()
		
		volume1 = volume1.view(batch_size*num_output_features, volume_size, volume_size, volume_size)
		volume2 = volume2.view(batch_size*num_output_features, volume_size, volume_size, volume_size)
		
		input_volume1 = F.pad(volume1, (0, volume_size, 0, volume_size, 0, volume_size)).contiguous()
		input_volume2 = F.pad(volume2, (0, volume_size, 0, volume_size, 0, volume_size)).contiguous()
		
		circ_volume = CrossConvFunction.apply(input_volume1, input_volume2)
		
		output_volume_size = circ_volume.size(2)
		volume = SwapQuadrants3DFunction.apply(circ_volume)

		return volume.view(batch_size, num_output_features, output_volume_size, output_volume_size, output_volume_size)