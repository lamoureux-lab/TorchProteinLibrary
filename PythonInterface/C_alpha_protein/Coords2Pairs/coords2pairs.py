import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppCoords2Pairs
import math
class Coords2PairsFunction(Function):
	"""
	Protein coordinates -> pairwise coordinates function
	"""
	def __init__(self, angles_max_length = 60):
		super(Coords2PairsFunction, self).__init__()
		self.angles_max_length = angles_max_length
		self.max_num_atoms = self.angles_max_length + 1
				
	def forward(self, input, angles_length):
		output_pairs_gpu = None
		if len(input.size())==1:
			#allocating output on gpu
			output_pairs_gpu = torch.FloatTensor(3*self.max_num_atoms*self.max_num_atoms).cuda()
			
		elif len(input.size())==2:
			#allocating output on gpu
			batch_size = input.size()[0]
			output_pairs_gpu = torch.FloatTensor(batch_size, 3*self.max_num_atoms*self.max_num_atoms).cuda()
			
		else:
			raise ValueError('Coords2PairsFunction: ', 'Incorrect input size:', input.size()) 
		
		output_pairs_gpu.fill_(0.0)
		
		cppCoords2Pairs.Coords2Pairs_forward( 	input,              #input coordinates
												output_pairs_gpu,  #output pairwise coordinates
												angles_length)
		
		self.save_for_backward(input, angles_length)

		if math.isnan(output_pairs_gpu.sum()):
			print 'Input: ', input
			print 'Output: ', output_pairs_gpu
			raise(Exception('Coords2PairsFunction: forward Nan'))		
		

		return output_pairs_gpu
			

	def backward(self, gradOutput):
		input, angles_length = self.saved_tensors
		if len(input.size()) == 1:
			gradInput_gpu = torch.FloatTensor(3*self.max_num_atoms).cuda()
		if len(input.size()) == 2:
			batch_size = input.size()[0]
			gradInput_gpu = torch.FloatTensor(batch_size, 3*self.max_num_atoms).cuda()
		gradInput_gpu.fill_(0.0)
		cppCoords2Pairs.Coords2Pairs_backward(gradInput_gpu, gradOutput, angles_length)
		
		if math.isnan(gradInput_gpu.sum()):
			print 'GradInput: ', gradInput_gpu
			print 'GradOutput: ', gradOutput
			raise(Exception('Coords2PairsFunction: backward Nan'))		
		
		return gradInput_gpu, None

class Coords2Pairs(Module):
	def __init__(self, angles_max_length):
		super(Coords2Pairs, self).__init__()
		self.angles_max_length = angles_max_length

	def forward(self, input, angles_length):
		return Coords2PairsFunction(self.angles_max_length)(input, angles_length)