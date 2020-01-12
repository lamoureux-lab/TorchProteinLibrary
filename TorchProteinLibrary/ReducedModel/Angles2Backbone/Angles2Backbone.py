import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _ReducedModel
import math

class Angles2BackboneGPUFunction(Function):
	"""
	Backbone angles -> backbone coordinates function
	"""
	@staticmethod
	def forward(ctx, input_angles, param, angles_length):
		ctx.save_for_backward(input_angles, param, angles_length)
		batch_size = input_angles.size(0)
		angles_max_length = input_angles.size(2)
		atoms_max_length = 3*angles_max_length

		if len(input_angles.size())==3:
			output_coords_gpu = torch.zeros(batch_size, 3*atoms_max_length, dtype=torch.float, device='cuda')
			ctx.A = torch.zeros(batch_size, 16*atoms_max_length, dtype=torch.float, device='cuda')
		else:
			raise Exception('Angles2BackboneFunction: ', 'Incorrect input size:', input_angles.size()) 

		_ReducedModel.Angles2BackboneGPU_forward( input_angles, param, output_coords_gpu, angles_length, ctx.A)
													
		if math.isnan(output_coords_gpu.sum()):
			raise(Exception('Angles2BackboneFunction: forward Nan'))

		return output_coords_gpu
			
	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		input_angles, param, angles_length = ctx.saved_tensors
		batch_size = input_angles.size(0)
		angles_max_length = input_angles.size(2)
		atoms_max_length = 3*angles_max_length

		gradInput_gpu = None
		gradParam_gpu = None
		if input_angles.requires_grad:
			gradInput_gpu = torch.zeros(batch_size, 3, angles_max_length, dtype=torch.float, device='cuda')
			dr_dangle = torch.zeros(batch_size, 3, 3*atoms_max_length*angles_max_length, dtype=torch.float, device='cuda')	
			_ReducedModel.Angles2BackboneGPUAngles_backward(gradInput_gpu, gradOutput, input_angles, param, angles_length, 
														ctx.A, dr_dangle)
		if param.requires_grad:
			gradParam_gpu = torch.zeros(6, dtype=torch.float, device='cuda')
			dr_dparam = torch.zeros(batch_size, 6, 3*atoms_max_length*angles_max_length, dtype=torch.float, device='cuda')
			_ReducedModel.Angles2BackboneGPUParam_backward(gradParam_gpu, gradOutput, input_angles, param, angles_length, 
														ctx.A, dr_dparam)
		
		# if math.isnan(torch.sum(gradInput_gpu)):
		# 	raise(Exception('Angles2BackboneFunction: backward Nan'))		
		
		return gradInput_gpu, gradParam_gpu, None

class Angles2BackboneCPUFunction(Function):
	"""
	Backbone angles -> backbone coordinates function
	"""
	@staticmethod
	def forward(ctx, input, param, angles_length):
		ctx.angles_max_length = input.size(2)
		ctx.atoms_max_length = 3*ctx.angles_max_length
		if len(input.size())==3:
			batch_size = input.size(0)
			output_coords_cpu = torch.zeros(batch_size, 3*(ctx.atoms_max_length), dtype=torch.double, device='cpu')
			ctx.A = torch.zeros(batch_size, 16*ctx.atoms_max_length, dtype=torch.double, device='cpu')
		else:
			raise Exception('Angles2BackboneFunction: ', 'Incorrect input size:', input.size()) 
		
		_ReducedModel.Angles2BackboneCPU_forward( input, param, output_coords_cpu, angles_length, ctx.A)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Angles2BackboneFunction: output_coords_cpu forward Nan'))

		ctx.save_for_backward(input, param, angles_length)
		return output_coords_cpu
			
	@staticmethod
	def backward(ctx, gradOutput_cpu):
		gradOutput_cpu = gradOutput_cpu.contiguous()
		input_angles, param, angles_length = ctx.saved_tensors
		batch_size = input_angles.size(0)
		angles_max_length = input_angles.size(2)
		atoms_max_length = 3*angles_max_length
		
		gradInput_gpu = None
		gradParam_gpu = None
		if input_angles.requires_grad:
			gradInput_cpu = torch.zeros(batch_size, 3, angles_max_length, dtype=torch.double, device='cpu')
			dr_dangle = torch.zeros(batch_size, 3, 3*atoms_max_length*angles_max_length, dtype=torch.double, device='cpu')
			_ReducedModel.Angles2BackboneCPUAngles_backward(gradInput_cpu, gradOutput_cpu, input_angles, param, angles_length, 
														ctx.A, dr_dangle)
		if param.requires_grad:
			gradParam_cpu = torch.zeros(6, dtype=torch.double, device='cpu')
			dr_dparam = torch.zeros(batch_size, 6, 3*atoms_max_length*angles_max_length, dtype=torch.double, device='cpu')
			_ReducedModel.Angles2BackboneCPUParam_backward(gradParam_cpu, gradOutput_cpu, input_angles, param, angles_length, 
														ctx.A, dr_dparam)

		
		# if math.isnan(torch.sum(gradInput_cpu)):
		# 	raise(Exception('Angles2BackboneFunction: gradInput_cpu backward Nan'))		
		
		return gradInput_cpu, gradParam_cpu, None


class Angles2Backbone(Module):
	def __init__(self):
		super(Angles2Backbone, self).__init__()
		
	def _fill_default_params(self, param):
		self.param2idx = {
			'R_N_CA': 0,
			'C_N_CA': 1,
			'R_CA_C': 2,
			'N_CA_C': 3,
			'R_C_N': 4,
			'CA_C_N': 5
		}
		self.idx2param = {v: k for k, v in self.param2idx.items()}

		param.data[self.param2idx['R_CA_C']] =  1.525
		param.data[self.param2idx['R_C_N']] =  1.330
		param.data[self.param2idx['R_N_CA']] =  1.460
		param.data[self.param2idx['CA_C_N']] =  math.pi - 2.1186
		param.data[self.param2idx['C_N_CA']] =  math.pi - 1.9391
		param.data[self.param2idx['N_CA_C']] =  math.pi - 2.061
		return param
	
	def get_default_parameters(self):
		param = torch.zeros(6, dtype=torch.double, device='cpu')
		param = self._fill_default_params(param)
		return param

	def forward(self, input, param, angles_length):
		if input.is_cuda:
			return Angles2BackboneGPUFunction.apply(input, param, angles_length)
		else:
			return Angles2BackboneCPUFunction.apply(input, param, angles_length)
