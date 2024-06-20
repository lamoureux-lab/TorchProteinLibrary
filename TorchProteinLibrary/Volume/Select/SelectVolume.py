import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Volume
import math

class CoordSelectFunction(Function):
	"""
	Atomic coords -> selected volume cells
	"""
	# @profile	
	@staticmethod
	def forward(ctx, volume_gpu, input_coords_gpu, num_atoms, resolution):
		ctx.resolution = resolution
		ctx.save_for_backward(input_coords_gpu, num_atoms, volume_gpu)

		if len(volume_gpu.size())==5 and len(input_coords_gpu.size())==2:
			batch_size = volume_gpu.size(0)
			num_features = volume_gpu.size(1)
			max_num_atoms = int(input_coords_gpu.size(1)/3)
			features = torch.zeros(batch_size, num_features, max_num_atoms, dtype=torch.float, device='cuda')
		else:
			raise(Exception("SelectVolume: Wrong input sizes", volume_gpu.size(), input_coords_gpu.size()))
		
		_Volume.SelectVolume_forward(volume_gpu, input_coords_gpu, num_atoms, features, ctx.resolution)

		return features
	
	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		input_coords_gpu, num_atoms, volume_gpu = ctx.saved_tensors
		gradInput = torch.zeros_like(volume_gpu)

		_Volume.SelectVolume_backward(gradOutput, gradInput, input_coords_gpu, num_atoms, ctx.resolution)
		
		return gradInput, None, None, None, None

class CoordsSelect(Module):
	def __init__(self, box_size_bins=120, box_size_ang=120):
		super(CoordsSelect, self).__init__()
		self.box_size_bins = box_size_bins
		self.box_size_ang = box_size_ang
		self.resolution = float(box_size_ang)/float(box_size_bins)
				
	def forward(self, volume, coords, num_atoms):
		return CoordSelectFunction.apply(volume, coords, num_atoms, self.resolution)

		
