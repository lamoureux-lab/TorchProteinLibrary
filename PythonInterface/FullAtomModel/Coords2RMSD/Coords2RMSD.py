import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppCoords2RMSD
import math

class Coords2RMSDFunction(Function):
	"""
	Protein coords, target coords -> rmsd function
	"""
	def __init__(self, max_num_atoms = 60):
		super(Coords2RMSDFunction, self).__init__()
		self.max_num_atoms = max_num_atoms
				
		self.c_coords_input = None
		self.c_coords_target = None
		self.U_coordinates_src = None
		self.Ut_coordinates_dst = None
		
		self.num_atoms = None
		
		
	def forward(self, input, target, num_atoms):
		# self.save_for_backward(angles_length)
		if self.num_atoms is None:
			self.num_atoms = torch.IntTensor(num_atoms.size(0)).copy_(num_atoms)
		
		if len(input.size())==2:
			batch_size = input.size()[0]
			output = torch.DoubleTensor(batch_size)
			#allocating temp outputs on gpu
			if self.c_coords_input is None:
				self.c_coords_input = torch.DoubleTensor(batch_size, 3*self.max_num_atoms)
			if self.c_coords_target is None:
				self.c_coords_target = torch.DoubleTensor(batch_size, 3*self.max_num_atoms)
			if self.U_coordinates_src is None:
				self.U_coordinates_src = torch.DoubleTensor(batch_size, 3*self.max_num_atoms)
			if self.Ut_coordinates_dst is None:
				self.Ut_coordinates_dst = torch.DoubleTensor(batch_size, 3*self.max_num_atoms)
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', input.size())

		cppCoords2RMSD.Coords2RMSD_forward( input, target, output,
											self.c_coords_input,
											self.c_coords_target,
											self.U_coordinates_src,
											self.Ut_coordinates_dst,
											self.num_atoms)
		
		self.save_for_backward(output)
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSDFunction: forward Nan'))
		
		return output
			

	def backward(self, gradOutput):
		if len(self.c_coords_input.size()) == 2:
			batch_size = self.c_coords_input.size()[0]
			gradInput = torch.DoubleTensor(batch_size, 3*self.max_num_atoms)
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', self.c_coords_input.size())
		
		gradInput.fill_(0.0)
		cppCoords2RMSD.Coords2RMSD_backward(gradInput, gradOutput,
											self.c_coords_input,
											self.c_coords_target,
											self.U_coordinates_src,
											self.Ut_coordinates_dst,
											self.num_atoms
											)
		
		output, = self.saved_tensors
		for i in range(batch_size):
			gradInput[i,:] = gradInput[i,:]/math.sqrt(output[i])
		
		if math.isnan(gradInput.sum()):
			raise(Exception('Coords2RMSDFunction: backward Nan'))	
		
		return gradInput, None, None


class Coords2RMSD(Module):
	def __init__(self, max_num_atoms):
		super(Coords2RMSD, self).__init__()
		self.max_num_atoms = max_num_atoms

	def forward(self, input, target, num_atoms):
		return Coords2RMSDFunction(self.max_num_atoms)(input, target, num_atoms)
