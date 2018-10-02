import sys
import os
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea

import torch
import torch.nn as nn
from torch.autograd import Variable

from TorchProteinLibrary.Volume import VolumeConvolution

def fill_exp(r0, volume):
	volume_size = volume.size(2)
	for x in range(volume_size):
		for y in range(volume_size):
			for z in range(volume_size):
				r = np.array([x,y,z]) - r0
				dist = np.linalg.norm(r)
				volume.data[0,0,x,y,z] = np.exp(-0.5*dist*dist)

def fill_delta(r0, volume):
	volume.data.fill_(0.0)
	volume.data[0,0,r0[0],r0[1],r0[2]] = 1

def get_argmax(volume):
	volume_size = volume.size(0)
	arg = (0,0,0)
	max_val = volume.data[0,0,0]
	
	for x in range(volume_size):
		for y in range(volume_size):
			for z in range(volume_size):
				val = float(volume.data[x,y,z])
				if val>max_val:
					arg = (x,y,z)
					max_val = val

	return max_val, arg


if __name__=='__main__':
	vc = VolumeConvolution(num_features=1)
	vc.W.data.fill_(1.0)
	v_size = 30
	inp1 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	inp2 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	fill_delta(np.array([1,6,5]), inp1)
	fill_delta(np.array([5,5,5]), inp2)
	# print inp1
	# print inp2
	# plt.imshow(inp1[0,0,:,:,0].data.cpu().numpy())
	# plt.show()
	# plt.imshow(inp2[0,0,:,:,0].data.cpu().numpy())
	# plt.show()
	
	out = vc(inp1, inp2)
	print(get_argmax(out[0,:,:,:]))
	plt.imshow(out[0,:,:,1].data.cpu().numpy())
	plt.colorbar()
	# plt.show()

	# print out