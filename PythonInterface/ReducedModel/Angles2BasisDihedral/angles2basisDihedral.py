import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2BasisDihedral

class Angles2BasisDihedralFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	def __init__(self, angles_max_length = 60):
		super(Angles2BasisDihedralFunction, self).__init__()
		self.angles_max_length = angles_max_length
		self.needs_input_grad = (True, False)
		self.B = None
		
	def forward(self, input, angles_length):
		output_basis_gpu = None
		A = None
		if len(input.size())==2:
			#allocating output on gpu
			output_basis_gpu = torch.FloatTensor(3, 3*(self.angles_max_length+1)).fill_(0.0).cuda()
			if self.B is None:
				self.B = torch.FloatTensor(9*self.angles_max_length).fill_(0.0).cuda()
			
			
		elif len(input.size())==3:
			#allocating output on gpu
			batch_size = input.size()[0]
			output_basis_gpu = torch.FloatTensor(batch_size, 3, 3*(self.angles_max_length+1)).cuda()
			if self.B is None:
				self.B = torch.FloatTensor(batch_size, 9*self.angles_max_length).cuda()
			
		else:
			raise ValueError('Angles2BasisFunction: ', 'Incorrect input size:', input.size()) 
		output_basis_gpu.fill_(0.0)
		cppAngles2BasisDihedral.Angles2Basis_forward( input,              #input angles
												output_basis_gpu,  #output coordinates
												angles_length, 
												self.B)
		
		self.save_for_backward(input, angles_length)
		return output_basis_gpu
			

	def backward(self, gradOutput):
		input, angles_length = self.saved_tensors
		if len(input.size()) == 2:
			gradInput_gpu = torch.FloatTensor(2, self.angles_max_length).cuda()
			daxis_dangle = torch.FloatTensor(6, 3*(self.angles_max_length+1)*self.angles_max_length).cuda()
		if len(input.size()) == 3:
			batch_size = input.size()[0]
			gradInput_gpu = torch.FloatTensor(batch_size, 2, self.angles_max_length).cuda()
			daxis_dangle = torch.FloatTensor(batch_size, 6, 3*(self.angles_max_length+1)*self.angles_max_length).cuda()
		daxis_dangle.fill_(0.0)
		gradInput_gpu.fill_(0.0)
		cppAngles2BasisDihedral.Angles2Basis_backward(gradInput_gpu, gradOutput, input, angles_length, self.B, daxis_dangle)

		return gradInput_gpu, None

class Angles2BasisDihedral(Module):
	def __init__(self, angles_max_length):
		super(Angles2BasisDihedral, self).__init__()
		self.angles_max_length = angles_max_length

	def forward(self, input, angles_length):
		return Angles2BasisDihedralFunction(self.angles_max_length)(input, angles_length)