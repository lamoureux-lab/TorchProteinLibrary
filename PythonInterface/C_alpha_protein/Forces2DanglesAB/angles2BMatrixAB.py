import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2BMatrixAB
import math

class Angles2BMatrixFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	def __init__(self, angles_max_length = 60):
		super(Angles2BMatrixFunction, self).__init__()
		self.angles_max_length = angles_max_length
				
	def forward(self, input_angles, input_coords, angles_length):
		output_B_gpu = None
				
		if len(input_angles.size())==2:
			#allocating output on gpu
			output_B_gpu = torch.FloatTensor(2,3*(self.angles_max_length)*(self.angles_max_length+1)).cuda()
						
		elif len(input_angles.size())==3:
			#allocating output on gpu
			batch_size = input_angles.size()[0]
			output_B_gpu = torch.FloatTensor(batch_size, 2, 3*(self.angles_max_length)*(self.angles_max_length+1)).cuda()
		else:
			raise ValueError('Angles2BMatrixFunction: ', 'Incorrect input size:', input_angles.size()) 
		output_B_gpu.fill_(0.0)
		cppAngles2BMatrixAB.Angles2BMatrix_forward( input_angles,              #input angles
													input_coords,              #input angles
													output_B_gpu,  #output coordinates
													angles_length)
										
		if math.isnan(output_B_gpu.sum()):
			raise(Exception('Angles2BMatrixFunction: forward Nan'))
		# self.save_for_backward(input, angles_length)
		return output_B_gpu
			

	def backward(self, gradOutput):
		pass
		

class Angles2BMatrixAB(Module):
	def __init__(self, angles_max_length):
		super(Angles2BMatrixAB, self).__init__()
		self.angles_max_length = angles_max_length

	def forward(self, input_angles, input_coords, angles_length):
		return Angles2BMatrixFunction(self.angles_max_length)(input_angles, input_coords, angles_length)