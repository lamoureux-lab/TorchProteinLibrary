import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _RMSD_CPU
import _RMSD_GPU
import math


class Coords2RMSD_GPU_Function(Function):
	"""
	Protein coords, target coords -> rmsd function
	"""

	@staticmethod		
	def forward(ctx, input, target, num_atoms):
				
		max_atoms = torch.max(num_atoms)
		if len(input.size())==2:
			batch_size = input.size(0)
			output = torch.DoubleTensor(batch_size).cuda()
			ctx.Ut_coordinates_dst = torch.zeros(batch_size, 3*max_atoms, dtype=torch.double, device='cuda')
		else:
			raise ValueError('Coords2RMSD_GPU_Function: ', 'Incorrect input size:', input.size())
		
		re_input = torch.zeros(input.size(), dtype=torch.double, device='cuda').copy_(input.double())
		re_target = torch.zeros(target.size(), dtype=torch.double, device='cuda').copy_(target.double())
		
		re_input.resize_(batch_size, max_atoms, 3)
		re_target.resize_(batch_size, max_atoms, 3)
		center_input = re_input.sum(dim=1)/num_atoms.unsqueeze(dim=1).double()
		center_target = re_target.sum(dim=1)/num_atoms.unsqueeze(dim=1).double()
		ctx.c_coords_input = (re_input - center_input.unsqueeze(dim=1)).resize_(batch_size, max_atoms*3).contiguous()
		ctx.c_coords_target = (re_target - center_target.unsqueeze(dim=1)).resize_(batch_size, max_atoms*3).contiguous()
		
		
		_RMSD_GPU.Coords2RMSD_GPU_forward( 	ctx.c_coords_input, ctx.c_coords_target, 
											output, num_atoms, ctx.Ut_coordinates_dst)
		
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSD_GPU_Function: forward Nan'))
		
		ctx.save_for_backward(output, num_atoms)
		return output
			
	@staticmethod
	def backward(ctx, gradOutput):
		output, num_atoms = ctx.saved_tensors
		max_atoms = torch.max(num_atoms)
		gradOutput = gradOutput.contiguous()

		if len(ctx.c_coords_input.size()) == 2:
			batch_size = ctx.c_coords_input.size(0)
			gradInput_gpu = torch.zeros(batch_size, 3*max_atoms, dtype=torch.double, device='cuda')
		else:
			raise ValueError('Coords2RMSD_GPU_Function: ', 'Incorrect input size:', gradOutput.size())
					
		gradInput_gpu = (ctx.c_coords_input - ctx.Ut_coordinates_dst)
		
		for i in range(batch_size):
			gradInput_gpu[i,:] = gradInput_gpu[i,:]/(math.sqrt(output[i]+1E-5)*num_atoms[i])
		
		if math.isnan(gradInput_gpu.sum()):
			raise(Exception('Coords2RMSD_GPU_Function: backward Nan'))		
		
		return Variable(gradInput_gpu), None, None


class Coords2RMSD_CPU_Function(Function):
	"""
	Protein coords, target coords -> rmsd function
	"""
	
	@staticmethod
	def forward(ctx, input, target, num_atoms):
		
		if len(input.size())==2:
			batch_size = input.size()[0]
			max_num_atoms = torch.max(num_atoms)
			#allocating temp outputs on cpu
			output = torch.zeros(batch_size, dtype=torch.double, device='cpu')
			ctx.c_coords_input = torch.zeros(batch_size, 3*max_num_atoms, dtype=torch.double, device='cpu')
			ctx.c_coords_target = torch.zeros(batch_size, 3*max_num_atoms, dtype=torch.double, device='cpu')
			ctx.U_coordinates_src = torch.zeros(batch_size, 3*max_num_atoms, dtype=torch.double, device='cpu')
			ctx.Ut_coordinates_dst = torch.zeros(batch_size, 3*max_num_atoms, dtype=torch.double, device='cpu')
		else:
			raise ValueError('Coords2RMSD_CPU_Function: ', 'Incorrect input size:', input.size())
		
		_RMSD_CPU.Coords2RMSD_CPU_forward( input, target, output,
											ctx.c_coords_input,
											ctx.c_coords_target,
											ctx.U_coordinates_src,
											ctx.Ut_coordinates_dst,
											num_atoms)
		
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSD_CPU_Function: forward Nan'))
		
		ctx.save_for_backward(output, num_atoms)
		return output
	
	@staticmethod
	def backward(ctx, gradOutput):

		output, num_atoms = ctx.saved_tensors
		
		if len(ctx.c_coords_input.size()) == 2:
			batch_size = ctx.c_coords_input.size(0)
			max_num_coords = ctx.c_coords_input.size(1)
			gradInput = torch.zeros(batch_size, max_num_coords, dtype=torch.double, device='cpu')
		else:
			raise ValueError('Coords2RMSD_CPU_Function: ', 'Incorrect input size:', c_coords_input.size())
				
		_RMSD_CPU.Coords2RMSD_CPU_backward(gradInput, gradOutput,
											ctx.c_coords_input,
											ctx.c_coords_target,
											ctx.U_coordinates_src,
											ctx.Ut_coordinates_dst,
											num_atoms
											)
		
		
		for i in range(batch_size):
			gradInput[i,:] = gradInput[i,:]/math.sqrt(output[i]+1E-5)
		
		if math.isnan(gradInput.sum()):
			raise(Exception('Coords2RMSD_CPU_Function: backward Nan'))	
		
		return Variable(gradInput), None, None


class Coords2RMSD(Module):
	def __init__(self):
		super(Coords2RMSD, self).__init__()
		
	def forward(self, input, target, num_atoms):
		if input.is_cuda:
			return Coords2RMSD_GPU_Function.apply(input, target, num_atoms)
		else:
			return Coords2RMSD_CPU_Function.apply(input, target, num_atoms)
