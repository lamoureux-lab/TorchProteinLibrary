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
	def __init__(self, angles_max_length = 60):
		super(Coords2RMSDFunction, self).__init__()
		self.angles_max_length = angles_max_length
		self.max_atoms = self.angles_max_length + 1
		self.needs_input_grad = (True, False, False)
		
		self.c_coords_input = None
		self.c_coords_target = None
		self.U_coordinates_src = None
		self.Ut_coordinates_dst = None
		self.rot_mat_t = None
		self.angles_length = None
		self.rmsd = None
		
	def forward(self, input, target, angles_length):
		# self.save_for_backward(angles_length)
		if self.angles_length is None:
			self.angles_length = torch.IntTensor(angles_length.size(0)).copy_(angles_length)
		if len(input.size())==1:
			output = torch.torch.FloatTensor(1).cuda()
			#allocating temp outputs
			if self.c_coords_input is None:
				self.c_coords_input = torch.FloatTensor(3*self.max_atoms).cuda()
			if self.c_coords_target is None:
				self.c_coords_target = torch.FloatTensor(3*self.max_atoms).cuda()
			if self.U_coordinates_src is None:
				self.U_coordinates_src = torch.FloatTensor(3*self.max_atoms).cuda()
			if self.Ut_coordinates_dst is None:
				self.Ut_coordinates_dst = torch.FloatTensor(3*self.max_atoms).cuda()
			if self.rot_mat_t is None:
				self.rot_mat_t = torch.FloatTensor(3,3).cuda()
						
		elif len(input.size())==2:
			batch_size = input.size()[0]
			output = torch.torch.FloatTensor(batch_size).cuda()
			#allocating temp outputs on gpu
			if self.c_coords_input is None:
				self.c_coords_input = torch.FloatTensor(batch_size, 3*self.max_atoms).cuda()
			if self.c_coords_target is None:
				self.c_coords_target = torch.FloatTensor(batch_size, 3*self.max_atoms).cuda()
			if self.U_coordinates_src is None:
				self.U_coordinates_src = torch.FloatTensor(batch_size, 3*self.max_atoms).cuda()
			if self.Ut_coordinates_dst is None:
				self.Ut_coordinates_dst = torch.FloatTensor(batch_size, 3*self.max_atoms).cuda()
			if self.rot_mat_t is None:
				self.rot_mat_t = torch.FloatTensor(batch_size, 3, 3).cuda()
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', input.size())
		# print 'Forward>'
		cppCoords2RMSD.Coords2RMSD_forward( input, target, output, angles_length,
											self.c_coords_input,
											self.c_coords_target,
											self.U_coordinates_src,
											self.Ut_coordinates_dst,
											self.rot_mat_t)
		# print '<Forward'
		# print 'input', torch.sum(input)
		# print 'target', torch.sum(target)
		# print torch.sum(self.c_coords_input)
		# print torch.sum(self.c_coords_target)
		# print 'output = ', output
		# print self.rot_mat_t
		self.save_for_backward(output)
		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSDFunction: forward Nan'))
		return output
			

	def backward(self, gradOutput):
		if len(self.c_coords_input.size()) == 1:
			gradInput_gpu = torch.FloatTensor(3*self.max_atoms).cuda()
					
		if len(self.c_coords_input.size()) == 2:
			batch_size = self.c_coords_input.size()[0]
			gradInput_gpu = torch.FloatTensor(batch_size, 3*self.max_atoms).cuda()
			
		gradInput_gpu.fill_(0.0)
		cppCoords2RMSD.Coords2RMSD_backward(gradInput_gpu, gradOutput, self.angles_length,
											self.c_coords_input,
											self.c_coords_target,
											self.Ut_coordinates_dst,
											self.rot_mat_t)
		
		output, = self.saved_tensors
		if len(self.c_coords_input.size()) == 1:
			gradInput_gpu = gradInput_gpu/math.sqrt(output[0])
		else:
			for i in range(batch_size):
				gradInput_gpu[i,:] = gradInput_gpu[i,:]/math.sqrt(output[i])
		if math.isnan(gradInput_gpu.sum()):
			raise(Exception('Coords2RMSDFunction: backward Nan'))		
		return gradInput_gpu, None, None

	def get_aligned_coordinates(self):
		# 
		proteins = []
		targets = []
		length = self.angles_length.squeeze()
		
		if length.size(0)>1:
			for i in range(0,length.size(0)):
				# print self.U_coordinates_src[i,:3*length[i]+3]
				# print self.c_coords_target[i,:3*length[i]+3]
				s_output = torch.FloatTensor(3*length[i]+3).fill_(0.0)
				# s_output.copy_(self.U_coordinates_src[i,:3*length[i]+3])
				s_output.copy_(self.c_coords_input[i,:3*length[i]+3])
				proteins.append(s_output.resize_(length[i]+1,3))

				s_target = torch.FloatTensor(3*length[i]+3).fill_(0.0)
				# s_target.copy_(self.c_coords_target[i,:3*length[i]+3])
				s_target.copy_(self.Ut_coordinates_dst[i,:3*length[i]+3])
				targets.append(s_target.resize_(length[i]+1,3))
		else:
			s_output = torch.FloatTensor(3*length[0]+3).fill_(0.0)
			# s_output.copy_(self.U_coordinates_src[:3*length[0]+3])
			s_output.copy_(self.c_coords_input[:3*length[0]+3])
			proteins.append(s_output.resize_(length[0]+1,3))
			
			s_target = torch.FloatTensor(3*length[0]+3).fill_(0.0)
			# s_target.copy_(self.c_coords_target[:3*length[0]+3])
			s_target.copy_(self.Ut_coordinates_dst[:3*length[0]+3])
			targets.append(s_target.resize_(length[0]+1,3))
		return proteins, targets


class Coords2RMSD(Module):
	def __init__(self, angles_max_length):
		super(Coords2RMSD, self).__init__()
		self.angles_max_length = angles_max_length

	def forward(self, input, target, angles_length):
		return Coords2RMSDFunction(self.angles_max_length)(input, target, angles_length)
