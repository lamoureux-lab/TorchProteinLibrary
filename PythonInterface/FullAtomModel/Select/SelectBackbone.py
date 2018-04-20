import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppSelect
import math

class SelectBackbone(Function):
	"""
	"""
		
	@staticmethod
	def forward(ctx, input_coords, input_atom_names, input_res_names, input_num_atoms):
		if not len(grad_output_coords.size()) == 2:
			raise ValueError('SelectBackbone: ', 'Incorrect input size:', input_coords.size())

		batch_size = input.size()[0]
		output_coords = torch.DoubleTensor(batch_size, input_coords.size(1)).zero_()
		output_atom_names = torch.ByteTensor(batch_size, input_atom_names.size(1), 4).zero_()
		output_res_names = torch.ByteTensor(batch_size, input_res_names.size(1), 4).zero_()
		output_num_atoms = torch.IntTensor(batch_size).zero_()
		backward_indexes = torch.IntTensor(batch_size, input_atom_names.size(1)).zero_()

		cppSelect.selectBB_forward( input_coords, input_atom_names, input_res_names, input_num_atoms
									output_coords, output_atom_names, output_res_names,	output_num_atoms,
									backward_indexes)

		ctx.backward_indexes = backward_indexes
		ctx.output_num_atoms = output_num_atoms
				
		if math.isnan(output_coords.sum()):
			raise(Exception('SelectBackbone: forward Nan'))
		
		return output
			
	@staticmethod
	def backward(ctx, grad_output_coords):

		if not len(grad_output_coords.size()) == 2:
			raise ValueError('SelectBackbone: ', 'Incorrect input size:', grad_output_coords.size())
		
		batch_size = ctx.c_coords_input.size(0)
		grad_input_coords = torch.DoubleTensor(batch_size, max_num_coords).zero_()
	
		cppSelect.selectBB_backward(grad_output_coords, grad_input_coords,
									ctx.output_num_atoms, ctx.backward_indexes)
									
		if math.isnan(grad_input_coords.sum()):
			raise(Exception('SelectBackbone: backward Nan'))	
		
		return Variable(grad_input_coords), None, None, None


class SelectBackbone(Module):
	def __init__(self):
		super(SelectBackbone, self).__init__()
		
	def forward(self, input_coords, input_atom_names, input_res_names, input_num_atoms):
		return SelectBackbone.apply(input_coords, input_atom_names, input_res_names, input_num_atoms)
