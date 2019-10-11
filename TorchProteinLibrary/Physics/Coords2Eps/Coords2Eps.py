import os
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Physics
import math
import numpy as np

class Coords2EpsFunction(Function):
	@staticmethod
	def forward(ctx, coords, assigned_params, num_atoms, box_size, resolution):
		coords = coords.contiguous()
		assigned_params = assigned_params.contiguous()
		num_atoms = num_atoms.contiguous()
		ctx.save_for_backward(coords, assigned_params, num_atoms)
		ctx.resolution = resolution

		batch_size = coords.size(0)
		rho_sum = torch.zeros(batch_size, box_size, box_size, box_size, dtype=torch.float, device='cuda')

		_Physics.Coords2Eps_forward(coords, assigned_params, num_atoms, rho_sum, resolution)

		return rho_sum

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		coords, assigned_params, num_atoms = ctx.saved_tensors
		
		num_coords = coords.size(1)
		gradInput = torch.zeros_like(coords)
		
		_Physics.Coords2Eps_backward(gradOutput, gradInput, coords, assigned_params, num_atoms)
		
		return gradInput, None, None, None, None, None

class Coords2Eps(Module):
	def __init__(self, box_size=80, resolution=1.0, eps_in=6.5, eps_out=79.0):
		super(Coords2Eps, self).__init__()
		self.eps_in = eps_in
		self.eps_out = eps_out
		self.box_size = box_size
		self.resolution = resolution
				
	def forward(self, coords, assigned_params, num_atoms):
		rho_sum = Coords2EpsFunction.apply(coords, assigned_params[:,:,1], num_atoms, self.box_size, self.resolution)
		eps = torch.exp(-rho_sum)*(self.eps_out - self.eps_in) + self.eps_in
		return eps