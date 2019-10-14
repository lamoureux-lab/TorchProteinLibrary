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
	def forward(ctx, coords, assigned_params, num_atoms, box_size, resolution, stern_size):
		coords = coords.contiguous()
		assigned_params = assigned_params.contiguous()
		num_atoms = num_atoms.contiguous()
		ctx.save_for_backward(coords, assigned_params, num_atoms)
		ctx.resolution = resolution

		batch_size = coords.size(0)
		rho_sum = torch.zeros(batch_size, 4, box_size, box_size, box_size, dtype=torch.float, device='cuda')
		_Physics.Coords2Eps_forward(coords, assigned_params, num_atoms, rho_sum, resolution, stern_size)

		return rho_sum

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		coords, assigned_params, num_atoms = ctx.saved_tensors
		
		num_coords = coords.size(1)
		gradInput = torch.zeros_like(coords)
		
		_Physics.Coords2Eps_backward(gradOutput, gradInput, coords, assigned_params, num_atoms)
		
		return gradInput, None, None, None, None, None

class Coords2QFunction(Function):
	@staticmethod
	def forward(ctx, coords, assigned_params, num_atoms, box_size, resolution):
		coords = coords.contiguous()
		assigned_params = assigned_params.contiguous()
		num_atoms = num_atoms.contiguous()
		ctx.save_for_backward(coords, assigned_params, num_atoms)
		ctx.resolution = resolution

		batch_size = coords.size(0)
		q_sum = torch.zeros(batch_size, box_size, box_size, box_size, dtype=torch.float, device='cuda')

		_Physics.Coords2Q_forward(coords, assigned_params, num_atoms, q_sum, resolution)

		return q_sum

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		coords, assigned_params, num_atoms = ctx.saved_tensors
		
		num_coords = coords.size(1)
		gradInput = torch.zeros_like(coords)
		
		_Physics.Coords2Eps_backward(gradOutput, gradInput, coords, assigned_params, num_atoms)
		
		return gradInput, None, None, None, None, None


class QEps2PhiFunction(Function):
	@staticmethod
	def forward(ctx, Q, Eps, resolution, kappa02):
		Q = Q.contiguous()
		Eps = Eps.contiguous()
		batch_size = Q.size(0)
		box_size = Q.size(1)
		
		Phi = torch.zeros(batch_size, box_size, box_size, box_size, dtype=torch.float, device='cuda')

		_Physics.QEps2Phi_forward(Q, Eps, Phi, resolution, kappa02)

		return Phi

	@staticmethod
	def backward(ctx, gradOutput):
		return None, None, None



class Coords2Elec(Module):
	def __init__(self, box_size=80, resolution=1.0, eps_in=6.5, eps_out=79.0, stern_size=1.0, kappa02=0.8486, charge_conv=7046.52):
		super(Coords2Elec, self).__init__()
		self.eps_in = eps_in
		self.eps_out = eps_out
		self.box_size = box_size
		self.resolution = resolution
		self.stern_size = stern_size
		self.kappa02 = kappa02
		self.charge_conv = charge_conv
				
	def forward(self, coords, assigned_params, num_atoms):
		rho_sum = Coords2EpsFunction.apply(coords, assigned_params[:,:,1], num_atoms, self.box_size, self.resolution, self.stern_size)
		eps = torch.exp(-rho_sum)*(self.eps_out - self.eps_in) + self.eps_in
		q = Coords2QFunction.apply(coords, assigned_params[:,:,0], num_atoms, self.box_size, self.resolution)
		q = q*(self.charge_conv)
		phi = QEps2PhiFunction.apply(q, eps, self.resolution, self.kappa02)

		return q, eps, phi