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

	@staticmethod		
	def forward(ctx, input, target, num_atoms):
				
		max_atoms = torch.max(num_atoms)
		if len(input.size())==2:
			batch_size = input.size(0)
			output = torch.DoubleTensor(batch_size).cuda()
			ctx.Ut_coordinates_dst = torch.DoubleTensor(batch_size, 3*max_atoms).cuda().zero_()
						
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', input.size())
		
		re_input = torch.DoubleTensor(input.size()).cuda().copy_(input.double())
		re_target = torch.DoubleTensor(target.size()).cuda().copy_(target.double())
		
		re_input.resize_(batch_size, max_atoms, 3)
		re_target.resize_(batch_size, max_atoms, 3)
		center_input = re_input.sum(dim=1)/num_atoms.unsqueeze(dim=1).double()
		center_target = re_target.sum(dim=1)/num_atoms.unsqueeze(dim=1).double()
		ctx.c_coords_input = (re_input - center_input.unsqueeze(dim=1)).resize_(batch_size, max_atoms*3).contiguous()
		ctx.c_coords_target = (re_target - center_target.unsqueeze(dim=1)).resize_(batch_size, max_atoms*3).contiguous()
		
		
		cppCoords2RMSD.Coords2RMSD_forward( ctx.c_coords_input, ctx.c_coords_target, 
											output, num_atoms, ctx.Ut_coordinates_dst)
		
		
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSDFunction: forward Nan'))
		
		ctx.save_for_backward(output, num_atoms)
		return output
			
	@staticmethod
	def backward(ctx, gradOutput):
		output, num_atoms = ctx.saved_tensors
		max_atoms = torch.max(num_atoms)
		gradOutput = gradOutput.contiguous()

		if len(ctx.c_coords_input.size()) == 2:
			batch_size = ctx.c_coords_input.size(0)
			gradInput_gpu = torch.DoubleTensor(batch_size, 3*max_atoms).cuda()
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', gradOutput.size())
					
		gradInput_gpu = (ctx.c_coords_input - ctx.Ut_coordinates_dst)
		
		for i in range(batch_size):
			gradInput_gpu[i,:] = gradInput_gpu[i,:]/(math.sqrt(output[i]+1E-5)*num_atoms[i])
		
		if math.isnan(gradInput_gpu.sum()):
			raise(Exception('Coords2RMSDFunction: backward Nan'))		
		
		return Variable(gradInput_gpu), None, None

class Coords2RMSD(Module):
	def __init__(self):
		super(Coords2RMSD, self).__init__()
		

	def forward(self, input, target, num_atoms):
		# self.num_atoms = num_atoms
		return Coords2RMSDFunction.apply(input, target, num_atoms)

	# def get_aligned_coordinates(self):
	# 	# 
	# 	proteins = []
	# 	targets = []
	# 	length = self.num_atoms.squeeze()
				
	# 	for i in xrange(length.size(0)):
	# 		s_output = torch.FloatTensor(3*length[i]+3).fill_(0.0)
	# 		s_output.copy_(self.c_coords_input[i,:3*length[i]+3])
	# 		proteins.append(s_output.resize_(length[i]+1,3))

	# 		s_target = torch.FloatTensor(3*length[i]+3).fill_(0.0)
	# 		s_target.copy_(self.Ut_coordinates_dst[i,:3*length[i]+3])
	# 		targets.append(s_target.resize_(length[i]+1,3))
	
	# 	return proteins, targets
