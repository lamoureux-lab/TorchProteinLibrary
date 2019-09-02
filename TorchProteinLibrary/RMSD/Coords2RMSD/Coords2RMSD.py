import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
#import _RMSD_CPU
#import _RMSD_GPU
import math
from TorchProteinLibrary.FullAtomModel.CoordsTransform import Coords2Center, CoordsTranslate
import _RMSD
import _FullAtomModel

class Coords2RMSDFunction(Function):
	"""
	Protein coords, target coords -> rmsd function
	"""

	@staticmethod		
	def forward(ctx, centered_input, centered_target, num_atoms):
		if len(centered_input.size())==2:
			batch_size = centered_input.size(0)
			num_coords = centered_input.size(1)
		else:
			raise ValueError('Coords2RMSDFunction: ', 'Incorrect input size:', input.size())

		if centered_input.is_cuda:
			output = torch.zeros(batch_size, dtype=centered_input.dtype, device='cuda')
			UT = torch.zeros(batch_size, 3, 3, dtype=centered_input.dtype, device='cuda')
			_RMSD.Coords2RMSDGPU_forward( centered_input, centered_target, output, num_atoms, UT)
		else:
			output = torch.zeros(batch_size, dtype=centered_input.dtype, device='cpu')
			UT = torch.zeros(batch_size, 3, 3, dtype=centered_input.dtype, device='cpu')
			_RMSD.Coords2RMSD_forward( centered_input, centered_target, output, num_atoms, UT)

		if math.isnan(output.sum()):
			raise(Exception('Coords2RMSDFunction: forward Nan'))
		
		ctx.save_for_backward(output, num_atoms, UT, centered_input, centered_target)
		return output, UT
			
	@staticmethod
	def backward(ctx, gradRMSD, gradUT):
		output, num_atoms, UT, centered_input, centered_target = ctx.saved_tensors
		gradRMSD = gradRMSD.contiguous()
		batch_size = centered_input.size(0)
		num_coords = centered_input.size(1)
		
		if centered_target.is_cuda:
			UT_centered_target = torch.zeros(batch_size, num_coords, dtype=centered_input.dtype, device='cuda')
			_FullAtomModel.CoordsRotateGPU_forward( centered_target, UT_centered_target, UT, num_atoms)
		else:
			UT_centered_target = torch.zeros(batch_size, num_coords, dtype=centered_input.dtype, device='cpu')
			_FullAtomModel.CoordsRotate_forward( centered_target, UT_centered_target, UT, num_atoms)
		
		gradInput_gpu = (centered_input - UT_centered_target)
		for i in range(batch_size):
			gradInput_gpu[i,:] = gradInput_gpu[i,:] * gradRMSD[i].item()/( float(output[i].item()) * float(num_atoms[i].item()) )
				
		
		if math.isnan(gradInput_gpu.sum()):
			raise(Exception('Coords2RMSDFunction: backward Nan'))		
		
		return gradInput_gpu, None, None

class Coords2RMSD(Module):
	def __init__(self):
		super(Coords2RMSD, self).__init__()
		self.c2c = Coords2Center()
		self.translate = CoordsTranslate()
		self.UT = None
		
	def forward(self, input, target, num_atoms):

		input_center = self.c2c(input, num_atoms)
		target_center = self.c2c(target, num_atoms)
		centered_input = self.translate(input, -input_center, num_atoms)
		centered_target = self.translate(target, -target_center, num_atoms)
		rmsd, self.UT = Coords2RMSDFunction.apply(centered_input, centered_target, num_atoms)
		return rmsd
