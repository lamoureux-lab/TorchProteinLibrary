import os
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Physics
import math
import numpy as np

class Coords2StressFunction(Function):
	@staticmethod
	def forward(ctx, coords, num_atoms, box_size, resolution):
		coords = coords.contiguous()
		num_atoms = num_atoms.contiguous()
		ctx.save_for_backward(coords, num_atoms)
		ctx.resolution = resolution

		batch_size = coords.size(0)
		

		return None

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		coords, num_atoms = ctx.saved_tensors
		
		num_coords = coords.size(1)
		gradInput = torch.zeros_like(coords)
						
		return None, None, None, None

class Coords2Stress(Module):
	def __init__(self, box_size=80, resolution=1.0):
		super(Coords2Stress, self).__init__()
			
				
	def forward(self, coords, num_atoms):

		
		pass