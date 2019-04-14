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
import _Volume

def fill_V1(r0, r1, R0, R1, volume):
	volume_size = volume.size(2)
	for x in range(volume_size):
		for y in range(volume_size):
			for z in range(volume_size):
				r_0 = np.array([x,y,z]) - r0
				r_1 = np.array([x,y,z]) - r1
				dist0 = np.linalg.norm(r_0)
				dist1 = np.linalg.norm(r_1)
				if dist0 < R0 and dist1 > R1:
					volume.data[0,0,x,y,z] = 1.0	

def fill_V2(r0, R0, volume):
	volume_size = volume.size(2)
	for x in range(volume_size):
		for y in range(volume_size):
			for z in range(volume_size):
				r_0 = np.array([x,y,z]) - r0
				dist0 = np.linalg.norm(r_0)
				if dist0 < R0:
					volume.data[0,0,x,y,z] = 1.0	

def get_boundary(volume_in, volume_out):
	volume_size = volume_in.size(2)
	for x in range(1,volume_size-1):
		for y in range(1,volume_size-1):
			for z in range(1,volume_size-1):
				cs = torch.sum(volume_in.data[0,0, x-1:x+2, y-1:y+2, z-1:z+2]).item()
				cc = volume_in.data[0,0,x,y,z].item()
				if (cc<0.01) and (cs>0.9):
					volume_out.data[0,0,x,y,z] = 1.0

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

def conv_numpy(inp1, inp2):
	npv1 = inp1.squeeze().cpu().numpy()
	npv2 = inp2.squeeze().cpu().numpy()
	cv1 = np.fft.fftn(npv1, [30,30,30])
	cv2 = np.fft.fftn(npv2, [30,30,30])
	cv = cv1 * np.conj(cv2)
	v = np.real(np.fft.ifftn(cv))
	return v

if __name__=='__main__':
	vc = VolumeConvolution()
	
	v_size = 30
	r0 = [15,15,15]
	r1 = [23,15,15]
	l0 = [15,20,15]

	inp1 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	inp1_border = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	inp2 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	out2 = torch.zeros(1, 1, v_size, v_size, v_size, dtype=torch.float, device='cuda')
	fill_V1(np.array(r0), np.array(r1), 8, 5, inp1)
	get_boundary(inp1, inp1_border)
	
	fill_V2(np.array(l0), 5, inp2)
	
	# v = conv_numpy(inp1, inp2)	
	# xyz = (0, 0, 0)
	# image1 = np.concatenate((v[xyz[0],:,:], v[:,xyz[1],:], v[:,:,xyz[2]]), axis = 1)
	# plt.imshow(image1)
	# plt.colorbar()
	# plt.show()

	_Volume.Volume2Xplor(inp1.squeeze().cpu(), "receptor.xplor", 1.0)
	_Volume.Volume2Xplor(inp1_border.squeeze().cpu(), "receptor_border.xplor", 1.0)
	_Volume.Volume2Xplor(inp2.squeeze().cpu(), "ligand.xplor", 1.0)
	
	border_overlap = vc(inp1_border, inp2)
	bulk_overlap = vc(inp1, inp2)
	print(bulk_overlap.size(), torch.max(bulk_overlap), torch.min(bulk_overlap))
	out = border_overlap - 0.5*bulk_overlap
	_Volume.Volume2Xplor(out.squeeze().cpu(), "out.xplor", 1.0)

	score, r = get_argmax(out[0,0,:,:,:])
	print(r)
	r = list(r)
	if r[0]>=v_size:
		r[0] = -(2*v_size - r[0])
	if r[1]>=v_size:
		r[1] = -(2*v_size - r[1])
	if r[2]>=v_size:
		r[2] = -(2*v_size - r[2])
	
	print(r)
	fill_V2(np.array([l0[0]+r[0],l0[1]+r[1],l0[2]+r[2]]), 5, out2)
	v_out = inp1 + out2
	_Volume.Volume2Xplor(v_out.squeeze().cpu(), "vout.xplor", 1.0)

	
	image = torch.cat([inp1[0,0,:,:,15], inp2[0,0,:,:,15]], dim=1).data.cpu().numpy()
	xyz = (0, 0, 0)
	image1 = torch.cat([out[0,0,xyz[0],:,:], out[0,0,:,xyz[1],:], out[0,0,:,:,xyz[2]]], dim=1).data.cpu().numpy()
	
	plt.subplot(211)
	plt.imshow(image)
	plt.colorbar()
	
	plt.subplot(212)
	plt.imshow(image1)
	plt.colorbar()
	# plt.show()
	plt.savefig("TestFig/test_forward.png")

	# print out