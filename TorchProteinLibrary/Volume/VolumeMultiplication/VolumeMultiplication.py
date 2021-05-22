import torch
from torch import nn
import numpy as np

class RzRxRz(nn.Module):
	"""
	Input: R:[phi, theta, psi]
	Output: RzRxRz Euler mat
	"""
	def __init__(self):
		super(RzRxRz, self).__init__()
	
	def forward(self, R):
		cphi = torch.cos(R[:, 0])
		sphi = torch.sin(R[:, 0])
		ctheta = torch.cos(R[:, 1])
		stheta = torch.sin(R[:, 1])
		cpsi = torch.cos(R[:, 2])
		spsi = torch.sin(R[:, 2])
				
		R0 = torch.stack([cphi*cpsi - spsi*ctheta*sphi, -cpsi*sphi - spsi*ctheta*cphi, spsi*stheta], dim=1)
		R1 = torch.stack([spsi*cphi + cpsi*ctheta*sphi, -spsi*sphi + cpsi*ctheta*cphi, -cpsi*stheta], dim=1)
		R2 = torch.stack([stheta*sphi, stheta*cphi, ctheta], dim=1)
		RotMat = torch.stack([R0, R1, R2], dim=2).transpose(1,2)

		return RotMat


class VolumeCrossMultiply(nn.Module):
	"""
	Input: R:[phi, theta, psi] T: [dx, dy, dz]
	Rotates around the volume center: [(box_size-1)/2,(box_size-1)/2,(box_size-1)/2]
	"""
	def __init__(self):
		super(VolumeCrossMultiply, self).__init__()
		self.rzrxrz = RzRxRz()
		self.transformed_volume = None

	def forward(self, volume1, volume2, R, T):
		batch_size = volume1.size(0)
		num_features = volume1.size(1)
		volume_size = volume1.size(2)
		mults = []

		perm = torch.tensor([2,1,0], dtype=torch.long, device=T.device)
		T = -2.0*T[:, perm]/volume_size

		R = self.rzrxrz(R)
		perm = torch.tensor([2,1,0], dtype=torch.long, device=R.device)
		R = R[:, :, perm]
		R = R[:, perm, :]

		RotMat = R.transpose(1, 2)
		TransMat = torch.stack([(RotMat[:,0,:]*T).sum(dim=1), (RotMat[:,1,:]*T).sum(dim=1), (RotMat[:,2,:]*T).sum(dim=1)], dim=1)
		# TransMat = T
		A = torch.cat([RotMat, TransMat.unsqueeze(dim=2)], dim=2)
		
		grid = nn.functional.affine_grid(A, size=volume2.size())
		volume2 = nn.functional.grid_sample(volume2, grid)
		self.transformed_volume = volume2

		volume1_unpacked = []
		volume2_unpacked = []
		for i in range(0, num_features):
	                volume1_unpacked.append(volume1)
			volume2_unpacked.append(volume2)
                        
		volume1 = torch.cat(volume1_unpacked, dim=1)
		volume2 = torch.cat(volume2_unpacked, dim=1)

		mults = (volume1 * volume2).sum(dim=[2,3,4])

		return mults
