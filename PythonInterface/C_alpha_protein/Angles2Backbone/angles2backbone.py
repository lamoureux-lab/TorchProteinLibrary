import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2Backbone
import math

class Angles2BackboneFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
	
	@staticmethod
	def forward(ctx, input, angles_length, norm):
		ctx.normalize = norm
		ctx.angles_max_length = torch.max(angles_length)
		ctx.atoms_max_length = 3*ctx.angles_max_length
		if len(input.size())==3:
			batch_size = input.size(0)
			output_coords_gpu = torch.FloatTensor(batch_size, 3*(ctx.atoms_max_length)).cuda()
			ctx.A = torch.FloatTensor(batch_size, 16*ctx.atoms_max_length).cuda()
		else:
			raise Exception('Angles2CoordsFunction: ', 'Incorrect input size:', input.size()) 

		output_coords_gpu.fill_(0.0)
		cppAngles2Backbone.Angles2Backbone_forward( input,              #input angles
												output_coords_gpu,  #output coordinates
												angles_length, 
												ctx.A)
													
		if math.isnan(output_coords_gpu.sum()):
			raise(Exception('Angles2BackboneFunction: forward Nan'))
		ctx.save_for_backward(input, angles_length)
		return output_coords_gpu
			
	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		input_angles, angles_length = ctx.saved_tensors
		if len(input_angles.size()) == 3:
			batch_size = input_angles.size(0)
			gradInput_gpu = torch.FloatTensor(batch_size, 2, ctx.angles_max_length).cuda()
			dr_dangle = torch.FloatTensor(batch_size, 2, 3*ctx.atoms_max_length*ctx.angles_max_length).cuda()
		else:
			raise(Exception('Angles2BackboneFunction: backward size', input_angles.size()))		
		
		dr_dangle.fill_(0.0)
		gradInput_gpu.fill_(0.0)
		
		cppAngles2Backbone.Angles2Backbone_backward(gradInput_gpu, gradOutput.data, input_angles, angles_length, ctx.A, dr_dangle, ctx.normalize)
		
		if math.isnan(torch.sum(gradInput_gpu)):
			print 'GradInput: ', gradInput_gpu
			print 'GradOutput: ', gradOutput
			raise(Exception('Angles2BackboneFunction: backward Nan'))		
		
		return Variable(gradInput_gpu), None, None

class Angles2Backbone(Module):
	def __init__(self, normalize = False):
		super(Angles2Backbone, self).__init__()
		self.normalize = normalize
		
	def forward(self, input, angles_length):
		return Angles2BackboneFunction.apply(input, angles_length, self.normalize)