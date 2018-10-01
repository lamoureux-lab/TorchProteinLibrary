import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Volume
import math

class SelectVolume:
	def __init__(self, box_size_bins=120, box_size_ang=120):
		self.box_size_bins = box_size_bins
		self.box_size_ang = box_size_ang
		self.resolution = float(box_size_ang)/float(box_size_bins)

	def __call__(self, volume, coords, num_atoms):
		if len(volume.size())==5 and len(coords.size())==2:
			batch_size = volume.size(0)
			num_features = volume.size(1)
			max_num_atoms = int(coords.size(1)/3)
			features = torch.zeros(batch_size, num_features, max_num_atoms, dtype=torch.float, device='cuda')
		else:
			raise(Exception("SelectVolume: Wrong input sizes", volume.size(), coords.size()))
		
		_Volume.SelectVolume_forward(volume, coords, num_atoms, features, self.resolution)

		return features
		
