import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim
from random import randint
from pairs2distributions import Pairs2Distributions

# def test_foward():
	
# 	max_atoms = 30
# 	max_angles = max_atoms-1
# 	num_atoms = 30
# 	num_angles = num_atoms-1
	
# 	num_types = 1
# 	num_bins = 10
# 	resolution = 1.0
# 	vectors = np.random.rand(num_atoms,3)*10
# 	pdist = torch.zeros(3*max_atoms*max_atoms)
# 	plane_stride = max_atoms*max_atoms
# 	for i in range(0,num_atoms):
# 		for j in range(0,num_atoms):
# 			pdist[i*max_atoms + j] = (vectors[i,0] - vectors[j,0])
# 			pdist[plane_stride + i*max_atoms + j] = (vectors[i,1] - vectors[j,1])
# 			pdist[2*plane_stride + i*max_atoms + j] = (vectors[i,2] - vectors[j,2])
			
# 	x0 = Variable(pdist.cuda(), requires_grad=True)
# 	length = Variable(torch.IntTensor(1).fill_(num_angles))
# 	types = Variable(torch.IntTensor(max_atoms).fill_(0).cuda())
	
# 	model = Pairs2Distributions(angles_max_length=max_angles, num_types=num_types, num_bins=num_bins, resolution=resolution).cuda()
		
# 	distr = model(x0, types, length)
# 	distr = distr.data
# 	x0 = x0.data
# 	types = types.data

# 	distr_man = np.zeros((num_atoms, num_types, num_bins, 3))
# 	distr = distr.resize_(num_atoms, num_types, num_bins, 3)
# 	for i in range(num_atoms):
# 		for j in range(num_atoms):
# 			if i==j: continue
# 			dist = np.linalg.norm(vectors[i,:] - vectors[j,:])
# 			xij = vectors[i,0]-vectors[j,0]
# 			yij = vectors[i,1]-vectors[j,1]
# 			zij = vectors[i,2]-vectors[j,2]
# 			if dist<0.001: continue
# 			rij = int(np.floor(dist/resolution))
# 			if rij>=num_bins: continue
# 			atype = 0
# 			distr_man[i,atype,rij,0] += xij / dist
# 			distr_man[i,atype,rij,1] += yij / dist
# 			distr_man[i,atype,rij,2] += zij / dist
	
# 	plt.imshow(distr_man.reshape((num_atoms,num_types*num_bins,3))[:,:,0])
# 	plt.show()

# 	distr_np = torch.FloatTensor(num_atoms,num_types,num_bins,3).copy_(distr).numpy()
# 	plt.imshow(distr_np.reshape((num_atoms,num_types*num_bins,3))[:,:,0])
# 	plt.show()
# 	print 'Error:', np.sum(np.abs(distr_man - distr_np))

def test_foward_nonvectorized():
	
	max_atoms = 30
	max_angles = max_atoms-1
	num_atoms = 30
	num_angles = num_atoms-1
	
	num_types = 3
	num_bins = 10
	resolution = 1.0
	vectors = np.random.rand(num_atoms,3)*10
	pdist = torch.zeros(3*max_atoms*max_atoms)
	plane_stride = max_atoms*max_atoms
	for i in range(0,num_atoms):
		for j in range(0,num_atoms):
			pdist[i*max_atoms + j] = (vectors[i,0] - vectors[j,0])
			pdist[plane_stride + i*max_atoms + j] = (vectors[i,1] - vectors[j,1])
			pdist[2*plane_stride + i*max_atoms + j] = (vectors[i,2] - vectors[j,2])
			
	x0 = Variable(pdist.cuda(), requires_grad=True)
	length = Variable(torch.IntTensor(1).fill_(num_angles))
	types = Variable(torch.IntTensor(max_atoms).fill_(0).cuda())
	for i in range(0,max_atoms):
		types[i] = randint(0, num_types-1)
	
	model = Pairs2Distributions(angles_max_length=max_angles, num_types=num_types, num_bins=num_bins, resolution=resolution).cuda()
		
	distr = model(x0, types, length)
	distr = distr.data
	x0 = x0.data
	types = types.data

	sigma = 2.0
	distr_man = np.zeros((num_atoms, num_types, num_bins))
	distr = distr.resize_(num_atoms, num_types, num_bins)
	for i in range(num_atoms):
		for j in range(num_atoms):
			if i==j: continue
			dist = np.linalg.norm(vectors[i,:] - vectors[j,:])
			atype = types[j]
			for k in range(num_bins):
				r = k*resolution
				distr_man[i,atype,k] += np.exp(-(r-dist)*(r-dist)/sigma)
	
	f, axarr = plt.subplots(2)
	axarr[0].imshow(distr_man.reshape((num_atoms,num_types*num_bins)))
	distr_np = torch.FloatTensor(num_atoms,num_types,num_bins).copy_(distr).numpy()
	axarr[1].imshow(distr_np.reshape((num_atoms,num_types*num_bins)))
	plt.savefig("TestFig/forward.png")
	# plt.show()
	print 'Error:', np.sum(np.abs(distr_man - distr_np))


if __name__=='__main__':
	test_foward_nonvectorized()