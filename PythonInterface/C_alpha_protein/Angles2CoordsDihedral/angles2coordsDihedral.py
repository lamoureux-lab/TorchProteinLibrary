import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2CoordsDihedral
import math

class Angles2CoordsDihedralFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	def __init__(self, angles_max_length = 60):
		super(Angles2CoordsDihedralFunction, self).__init__()
		self.angles_max_length = angles_max_length
		self.needs_input_grad = (True, False)
		self.A = None
		
	def forward(self, input, angles_length):
		output_coords_gpu = None
		A = None
		dr_dangle = None
		if len(input.size())==2:
			#allocating output on gpu
			output_coords_gpu = torch.FloatTensor(3*(self.angles_max_length+1)).cuda()
			if self.A is None:
				self.A = torch.FloatTensor(16*self.angles_max_length).cuda()
			
			
		elif len(input.size())==3:
			#allocating output on gpu
			batch_size = input.size()[0]
			output_coords_gpu = torch.FloatTensor(batch_size, 3*(self.angles_max_length+1)).cuda()
			if self.A is None:
				self.A = torch.FloatTensor(batch_size, 16*self.angles_max_length).cuda()
			
		else:
			raise ValueError('Angles2CoordsFunction: ', 'Incorrect input size:', input.size()) 
		output_coords_gpu.fill_(0.0)
		cppAngles2CoordsDihedral.Angles2Coords_forward( input,              #input angles
												output_coords_gpu,  #output coordinates
												angles_length, 
												self.A)
		# print 'A2C F:', output_coords_gpu.sum()												
		if math.isnan(output_coords_gpu.sum()):
			raise(Exception('Angles2CoordsDihedralFunction: forward Nan'))		
		self.save_for_backward(input, angles_length)
		return output_coords_gpu
			

	def backward(self, gradOutput):
		input, angles_length = self.saved_tensors
		if len(input.size()) == 2:
			gradInput_gpu = torch.FloatTensor(2, self.angles_max_length).cuda()
			dr_dangle = torch.FloatTensor(2, 3*(self.angles_max_length+1)*self.angles_max_length).cuda()
		if len(input.size()) == 3:
			batch_size = input.size()[0]
			gradInput_gpu = torch.FloatTensor(batch_size, 2, self.angles_max_length).cuda()
			dr_dangle = torch.FloatTensor(batch_size, 2, 3*(self.angles_max_length+1)*self.angles_max_length).cuda()
		dr_dangle.fill_(0.0)
		gradInput_gpu.fill_(0.0)
		
		cppAngles2CoordsDihedral.Angles2Coords_backward(gradInput_gpu, gradOutput, input, angles_length, self.A, dr_dangle)
		
		# print 'A2C gI:', gradInput_gpu.sum()
		if math.isnan(gradInput_gpu.sum()):
			print 'GradInput: ', gradInput_gpu
			print 'GradOutput: ', gradOutput
			raise(Exception('Angles2CoordsDihedralFunction: backward Nan'))		
		
		return gradInput_gpu, None

class Angles2CoordsDihedral(Module):
	def __init__(self, angles_max_length):
		super(Angles2CoordsDihedral, self).__init__()
		self.angles_max_length = angles_max_length

	def forward(self, input, angles_length):
		return Angles2CoordsDihedralFunction(self.angles_max_length)(input, angles_length)