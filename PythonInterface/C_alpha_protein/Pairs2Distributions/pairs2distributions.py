import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppPairs2Dist
import math
class Pairs2DistributionsFunction(Function):
	"""
	Protein pairwise coordinates -> pairwise distributions
	"""
	def __init__(self, angles_max_length = 60, num_types=21, num_bins=12, resolution=1.0):
		super(Pairs2DistributionsFunction, self).__init__()
		self.angles_max_length = angles_max_length
		self.max_num_atoms = self.angles_max_length + 1
		self.num_types = num_types
		self.num_bins = num_bins
		self.resolution = resolution
		# self.needs_input_grad = (True, False, False)
		
	def forward(self, input, input_types, angles_length):
		output_pairs_gpu = None
		if len(input.size())==1:
			#allocating output on gpu
			# output_dist_gpu = torch.FloatTensor(3*self.max_num_atoms*self.num_types*self.num_bins).cuda()
			output_dist_gpu = torch.FloatTensor(self.max_num_atoms*self.num_types*self.num_types*self.num_bins).cuda()
			
		elif len(input.size())==2:
			#allocating output on gpu
			batch_size = input.size()[0]
			
			# output_dist_gpu = torch.FloatTensor(batch_size, 3*self.max_num_atoms*self.num_types*self.num_bins).cuda()
			output_dist_gpu = torch.FloatTensor(batch_size, self.max_num_atoms*self.num_types*self.num_types*self.num_bins).cuda()
			
		else:
			raise ValueError('Pairs2DistributionsFunction: ', 'Incorrect input size:', input.size()) 
		
		output_dist_gpu.fill_(0.0)
		cppPairs2Dist.Pairs2Dist_forward( 	input,              #input coordinates
											output_dist_gpu,  #output pairwise coordinates
											input_types,
											angles_length,
											self.num_types,
											self.num_bins,
											self.resolution)
		
		if math.isnan(output_dist_gpu.sum()):
			raise(Exception('Pairs2DistributionsFunction: forward Nan'))

		self.save_for_backward(input, angles_length, input_types)
		return output_dist_gpu
			

	def backward(self, gradOutput):
		input, angles_length, input_types = self.saved_tensors
		if len(input.size()) == 1:
			gradInput_gpu = torch.FloatTensor(3*self.max_num_atoms*self.max_num_atoms).cuda()
	
		if len(input.size()) == 2:
			batch_size = input.size()[0]
			gradInput_gpu = torch.FloatTensor(batch_size, 3*self.max_num_atoms*self.max_num_atoms).cuda()
	
		gradInput_gpu.fill_(0.0)
		cppPairs2Dist.Pairs2Dist_backward(	gradInput_gpu,
											gradOutput,
											input, 
											input_types, 
											angles_length,
											self.num_types,
											self.num_bins,
											self.resolution)
		if math.isnan(gradInput_gpu.sum()):
			print gradOutput, gradOutput.sum()
			print gradInput_gpu, gradInput_gpu.sum()
			with open('pairs2distributions_backward_gradInput_gpu.pkl', 'w') as f:
				torch.save(gradInput_gpu, f)
			with open('pairs2distributions_backward_gradOutput.pkl', 'w') as f:
				torch.save(gradOutput, f)
			with open('pairs2distributions_backward_input.pkl', 'w') as f:
				torch.save(input, f)
			with open('pairs2distributions_backward_angles_length.pkl', 'w') as f:
				torch.save(angles_length, f)
			with open('pairs2distributions_backward_input_types.pkl', 'w') as f:
				torch.save(input_types, f)
			
			raise(Exception('Pairs2DistributionsFunction: backward Nan'))
		
		return gradInput_gpu, None, None

class Pairs2Distributions(Module):
	def __init__(self, angles_max_length, num_types=21, num_bins=12, resolution=1.0):
		super(Pairs2Distributions, self).__init__()
		self.angles_max_length = angles_max_length
		self.max_num_atoms = self.angles_max_length + 1
		self.num_types = num_types
		self.num_bins = num_bins
		self.resolution = resolution

	def forward(self, input, input_types, angles_length):
		return Pairs2DistributionsFunction(self.angles_max_length, self.num_types, self.num_bins, self.resolution)(input, input_types, angles_length)