import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _RMSD_CPU
import math

class Coords2RMSDFunction(Function):
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
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', input.size())
		
		_RMSD_CPU.Coords2RMSD_CPU_forward( input, target, output,
											ctx.c_coords_input,
											ctx.c_coords_target,
											ctx.U_coordinates_src,
											ctx.Ut_coordinates_dst,
											num_atoms)
		
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSDFunction: forward Nan'))
		
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
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', c_coords_input.size())
				
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
			raise(Exception('Coords2RMSDFunction: backward Nan'))	
		
		return Variable(gradInput), None, None


class Coords2RMSD(Module):
	def __init__(self):
		super(Coords2RMSD, self).__init__()
		
	def forward(self, input, target, num_atoms):
		return Coords2RMSDFunction.apply(input, target, num_atoms)
