import torch
from torch import nn
import numpy as np


class VolumeMultiply(nn.Module):
	def __init__(self):
		super(VolumeMultiply, self).__init__()

	def mabs(self, dx, dy, dz):
		return np.abs(dx), np.abs(dy), np.abs(dz)
	
	def multiply(self, v1, v2, dx, dy, dz, L):
		
		#all positive
		if dx>=0 and dy>=0 and dz>=0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,dx:L, dy:L, dz:L] * v2[:,0:L-dx, 0:L-dy, 0:L-dz]
		
		#one negative
		elif dx<0 and dy>=0 and dz>=0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,0:L-dx, dy:L, dz:L] * v2[:,dx:L, 0:L-dy, 0:L-dz]
		elif dx>=0 and dy<0 and dz>=0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,dx:L, 0:L-dy, dz:L] * v2[:,0:L-dx, dy:L, 0:L-dz]
		elif dx>=0 and dy>=0 and dz<0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,dx:L, dy:L, 0:L-dz] * v2[:,0:L-dx, 0:L-dy, dz:L]
		
		#one positive
		elif dx<0 and dy<0 and dz>=0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,0:L-dx, 0:L-dy, dz:L] * v2[:,dx:L, dy:L, 0:L-dz]
		elif dx>=0 and dy<0 and dz<0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,dx:L, 0:L-dy, 0:L-dz] * v2[:,0:L-dx, dy:L, dz:L]
		elif dx<0 and dy>=0 and dz<0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,0:L-dx, dy:L, 0:L-dz] * v2[:,dx:L, 0:L-dy, dz:L]
		
		#all negative
		elif dx<0 and dy<0 and dz<0:
			dx, dy, dz = self.mabs(dx, dy, dz)
			result = v1[:,0:L-dx, 0:L-dy, 0:L-dz] * v2[:,dx:L, dy:L, dz:L]

		return result.sum(dim=3).sum(dim=2).sum(dim=1).squeeze()

	def forward(self, receptor, ligand, T):
		batch_size = receptor.size(0)
		L = receptor.size(2)
		mults = []
		for i in range(batch_size):
			v1 = receptor[i,:,:,:,:].squeeze()
			v2 = ligand[i,:,:,:,:].squeeze()
			dx = int(T[i,0])
			dy = int(T[i,1])
			dz = int(T[i,2])
			mults.append(self.multiply(v1,v2,dx,dy,dz,L))
		return torch.stack(mults, dim=0)

class VolumeCrossMultiply(VolumeMultiply):
	def __init__(self):
		super(VolumeCrossMultiply, self).__init__()
	
	def forward(self, volume1, volume2, T):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		mults = []

		volume1_unpacked = []
		volume2_unpacked = []
		for i in range(0, num_features):
			volume1_unpacked.append(volume1[:,0:num_features-i,:,:,:])
			volume2_unpacked.append(volume2[:,i:num_features,:,:,:])
		volume1 = torch.cat(volume1_unpacked, dim=1)
		volume2 = torch.cat(volume2_unpacked, dim=1)

		for i in range(batch_size):
			v1 = volume1[i,:,:,:,:].squeeze()
			v2 = volume2[i,:,:,:,:].squeeze()
			dx = int(T[i,0].item())
			dy = int(T[i,1].item())
			dz = int(T[i,2].item())
			mults.append(self.multiply(v1,v2,dx,dy,dz,volume_size))
		return torch.stack(mults, dim=0)