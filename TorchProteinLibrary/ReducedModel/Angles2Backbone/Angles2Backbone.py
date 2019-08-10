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
	def forward(ctx, input_angles, angles_length):
		ctx.save_for_backward(input_angles, angles_length)
		batch_size = input_angles.size(0)
		angles_max_length = input_angles.size(2)
		atoms_max_length = 3*angles_max_length

		if len(input_angles.size())==3:
			output_coords_gpu = torch.zeros(batch_size, 3*atoms_max_length, dtype=torch.float, device='cuda')
			ctx.A = torch.zeros(batch_size, 16*atoms_max_length, dtype=torch.float, device='cuda')
		else:
			raise Exception('Angles2BackboneFunction: ', 'Incorrect input size:', input_angles.size()) 

		_ReducedModel.Angles2BackboneGPU_forward( input_angles, output_coords_gpu, angles_length, ctx.A)
													
		if math.isnan(output_coords_gpu.sum()):
			raise(Exception('Angles2BackboneFunction: forward Nan'))

		return output_coords_gpu
			
	@staticmethod
	def backward(ctx, gradOutput):
		input_angles, angles_length = ctx.saved_tensors
		batch_size = input_angles.size(0)
		angles_max_length = input_angles.size(2)
		atoms_max_length = 3*angles_max_length

		gradOutput = gradOutput.contiguous()
		
		if len(input_angles.size()) == 3:
			gradInput_gpu = torch.zeros(batch_size, 3, angles_max_length, dtype=torch.float, device='cuda')
			dr_dangle = torch.zeros(batch_size, 3, 3*atoms_max_length*angles_max_length, dtype=torch.float, device='cuda')
		else:
			raise(Exception('Angles2BackboneFunction: backward size', input_angles.size()))		
			
		_ReducedModel.Angles2BackboneGPU_backward(gradInput_gpu, gradOutput, input_angles, angles_length, ctx.A, dr_dangle)
		
		if math.isnan(torch.sum(gradInput_gpu)):
			raise(Exception('Angles2BackboneFunction: backward Nan'))		
		
		return gradInput_gpu, None, None

class Angles2BackboneCPUFunction(Function):
	"""
	Backbone angles -> backbone coordinates function
	"""
	@staticmethod
	def forward(ctx, input, angles_length):
		ctx.angles_max_length = input.size(2)
		ctx.atoms_max_length = 3*ctx.angles_max_length
		if len(input.size())==3:
			batch_size = input.size(0)
			output_coords_cpu = torch.zeros(batch_size, 3*(ctx.atoms_max_length), dtype=torch.double, device='cpu')
			ctx.A = torch.zeros(batch_size, 16*ctx.atoms_max_length, dtype=torch.double, device='cpu')
		else:
			raise Exception('Angles2BackboneFunction: ', 'Incorrect input size:', input.size()) 

		_ReducedModel.Angles2BackboneCPU_forward( input, output_coords_cpu, angles_length, ctx.A)

		if math.isnan(output_coords_cpu.sum()):
			raise(Exception('Angles2BackboneFunction: output_coords_cpu forward Nan'))

		ctx.save_for_backward(input, angles_length)
		return output_coords_cpu
			
	@staticmethod
	def backward(ctx, gradOutput_cpu):
		gradOutput_cpu = gradOutput_cpu.contiguous()
		input_angles, angles_length = ctx.saved_tensors
		if len(input_angles.size()) == 3:
			batch_size = input_angles.size(0)
			gradInput_cpu = torch.zeros(batch_size, 3, ctx.angles_max_length, dtype=torch.double, device='cpu')
			dr_dangle = torch.zeros(batch_size, 3, 3*ctx.atoms_max_length*ctx.angles_max_length, dtype=torch.double, device='cpu')
		else:
			raise(Exception('Angles2BackboneFunction: backward size', input_angles.size()))		
		
		_ReducedModel.Angles2BackboneCPU_backward(gradInput_cpu, gradOutput_cpu, input_angles, angles_length, ctx.A, dr_dangle)
		
		if math.isnan(torch.sum(gradInput_cpu)):
			raise(Exception('Angles2BackboneFunction: gradInput_cpu backward Nan'))		
		
		return gradInput_cpu, None, None


class Angles2Backbone(Module):
	def __init__(self):
		super(Angles2Backbone, self).__init__()
				
	def forward(self, input, angles_length):
		if input.is_cuda:
			return Angles2BackboneGPUFunction.apply(input, angles_length)
		else:
			return Angles2BackboneCPUFunction.apply(input, angles_length)