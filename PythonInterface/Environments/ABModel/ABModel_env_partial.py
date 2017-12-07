"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from spaces import TorchContinuous
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from C_alpha_protein.Angles2CoordsAB import cppAngles2CoordsAB
from C_alpha_protein.Coords2Pairs import cppCoords2Pairs
from C_alpha_protein.Forces2DanglesAB import cppForces2DanglesAB
from C_alpha_protein.Ddist2Forces import cppDdist2Forces

class ABModelEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self):
		# self.sequence = 'ABBABBABABBAB'
		self.sequence = 'ABB'
		self.num_bins = 20
		self.resolution = 0.2
		self.num_types = 2
		self.angles_length = len(self.sequence)-1
		self.num_atoms = len(self.sequence)
		self.beta = 0.1
		self.num_steps = 10
		
		self.batch_size = 1
		self.angles = torch.squeeze(torch.FloatTensor(self.batch_size, 2, self.angles_length).cuda())
		self.A = torch.squeeze(torch.FloatTensor(self.batch_size, 16*self.angles_length).cuda())
		self.angles_length_tensor = torch.IntTensor(self.batch_size).fill_(self.angles_length)
		self.coords = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.new_coords = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		
		self.pairs = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms*self.num_atoms).cuda())

		self.B = torch.squeeze(torch.FloatTensor(self.batch_size, 2, 3*self.angles_length*self.num_atoms).cuda())
		self.forces = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.dangles = torch.squeeze(torch.FloatTensor(self.batch_size, 2, self.angles_length).cuda())
				
		
		self.mask = torch.ByteTensor(self.batch_size, self.num_atoms*self.num_atoms).cuda().fill_(0)
		self.mask[:, ::self.num_atoms+1]=1
		self.mask[:, 1::self.num_atoms+1]=1
		self.mask[:, self.num_atoms::self.num_atoms+1]=1
		self.mask.resize_(self.batch_size, self.num_atoms, self.num_atoms)
		self.mask = torch.squeeze(self.mask)
		
		self.prev_energy = torch.FloatTensor(self.batch_size).fill_(float('+inf')).cuda()
		
		self.seq_int = torch.IntTensor(self.batch_size, self.num_atoms).cuda()
		self.interaction = torch.FloatTensor(self.batch_size, self.num_atoms, self.num_atoms).cuda()
		for i in xrange(self.num_atoms):
			if self.sequence[i] == 'A':
				self.seq_int[:,i] = 1
			else:
				self.seq_int[:,i] = 0

			for j in xrange(self.num_atoms):
				if self.sequence[i] == 'A' and self.sequence[j] == 'A':
					self.interaction[:,i,j].fill_(1.0)
				elif self.sequence[i] == 'B' and self.sequence[j] == 'B':
					self.interaction[:,i,j].fill_(0.5)
				else:
					self.interaction[:,i,j].fill_(-0.5)

		self.seq_int = torch.squeeze(self.seq_int)
		self.interaction = torch.squeeze(self.interaction)
		
		self.action_space = TorchContinuous(low=float('-inf'), high=float('+inf'), shape = (self.num_atoms, self.num_atoms))
		self.observation_space = TorchContinuous(low=float('-inf'), high=float('+inf'), shape = (self.num_bins, self.num_atoms, self.num_atoms))

		self.iterations = 0

		self.ones = torch.FloatTensor(self.batch_size).cuda().fill_(1.0)

		# self.compute_internal_variables(self.action_space.sample(), self.coords)


	def compute_internal_variables(self, angles, coords):

		coords.fill_(0.0)
		cppAngles2CoordsAB.Angles2Coords_forward(   angles,  #input angles
													coords,  #output coordinates
													self.angles_length_tensor, 
													self.A)
		if math.isnan(coords.sum()):
			raise(Exception('ABModel: angles2coords forward Nan'))		

		self.pairs.fill_(0.0)
		cppCoords2Pairs.Coords2Pairs_forward( 	coords, #input coordinates
												self.pairs,  #output pairwise coordinates
												self.angles_length_tensor)
		if math.isnan(self.pairs.sum()):
			raise(Exception('ABModel: angles2coords forward Nan'))
		
		self.pairs.resize_(self.batch_size, 3, self.num_atoms, self.num_atoms)
		dist = torch.pow( self.pairs[:,0,:,:], 2) + torch.pow( self.pairs[:,1,:,:], 2) + torch.pow( self.pairs[:,2,:,:], 2)
		self.pairs.resize_(self.batch_size, 3*self.num_atoms*self.num_atoms).squeeze_()

		observation = torch.FloatTensor(self.batch_size, self.num_bins, self.num_atoms, self.num_atoms).cuda()
		# observation[0,0,:,:] = self.interaction[:,:]

		for i in xrange(self.num_bins):
			r_low = i*self.resolution
			r_high = (i+1)*self.resolution
			observation[0,i, :, :] = torch.mul(torch.ge(dist[:,:],r_low), torch.le(dist[:,:], r_high))
			observation[0,i, :, :].mul_(self.interaction)

		self.obs = observation
		return torch.squeeze(observation)

	def compute_energy(self, angles):
		
		angles.resize_(self.batch_size, 2, self.angles_length)
		V1 = self.angles_length - torch.sum(torch.cos(angles[:,0,1:]), dim=1) - 1
		angles.squeeze_()
		
		self.pairs.resize_(self.batch_size, 3, self.num_atoms, self.num_atoms)
		r2 = torch.pow( self.pairs[:,0,:,:], 2) + torch.pow( self.pairs[:,1,:,:], 2) + torch.pow( self.pairs[:,2,:,:], 2)
		self.pairs.resize_(self.batch_size, 3*self.num_atoms*self.num_atoms).squeeze_()
		r2 = torch.clamp(r2, min=0.2)
		r6 = 1.0/torch.pow(r2, 3)
		r6.masked_fill_(self.mask, 0.0)
		r12 = torch.pow(r6, 2)
		r12 = torch.addcmul(r12, value=-1, tensor1=r6, tensor2=self.interaction)
		V2 =  torch.sum(torch.sum(r12,1),1)
		
		return 0.25*V1+2.0*V2
	
	def _step(self, action):
				
		self.forces.fill_(0.0)
		cppDdist2Forces.Ddist2Forces_forward(   	action,  
													self.coords,  
													self.forces,
													self.angles_length_tensor)
		if math.isnan(self.forces.sum()):
			raise(Exception('ABModel: ddist2forces forward Nan'))

		self.dangles.fill_(0.0)
		self.B.fill_(0.0)
		cppForces2DanglesAB.Forces2Dangles_forward( self.angles,
													self.coords,
													self.forces,
													self.B,
													self.dangles,
													self.angles_length_tensor)
		if math.isnan(self.dangles.sum()):
			raise(Exception('ABModel: forces2dangles forward Nan'))

		new_angles = self.angles + self.dangles
		observation = self.compute_internal_variables(new_angles, self.new_coords)
		new_energy = self.compute_energy(new_angles)
		
		if self.iterations>self.num_steps:
			done = True
		else:
			done = False

		self.iterations += 1

		# reward = torch.clamp(self.prev_energy - new_energy, -10.0, 10.0)
		# reward = self.prev_energy - new_energy
		if self.prev_energy[0]>new_energy[0]:
			reward = 1.0
		else:
			reward = -1.0
		self.angles.copy_(new_angles)
		self.prev_energy.copy_(new_energy)
		self.coords.copy_(self.new_coords)
		return observation, reward, done, {}


	def _reset(self):
		self.angles.uniform_(-0.1, 0.1)
		# self.angles.fill_(0.0)
		self.iterations = 0
		observation = self.compute_internal_variables(self.angles, self.coords)
		self.prev_energy = self.compute_energy(self.angles)
		return observation

	def _render(self, mode='human', close=False):
		import matplotlib as mpl
		mpl.use('Agg')
		import matplotlib.pylab as plt
		plt.ioff()
		import mpl_toolkits.mplot3d.axes3d as p3
		import seaborn as sea
		
		coordinates = self.coords.cpu()
		coordinates.resize_( self.num_atoms, 3)
		rx = coordinates[:self.num_atoms,0].numpy()
		ry = coordinates[:self.num_atoms,1].numpy()
		rz = coordinates[:self.num_atoms,2].numpy()
		
		
		fig = plt.figure(figsize=plt.figaspect(2.))
		ax = fig.add_subplot(2, 1, 1)
		obs_np = self.obs.cpu()
		size = self.obs.size(2)
		N = int(np.floor(np.sqrt(self.num_bins)))
		mat = np.zeros( (N*size, (N+1)*size) )
		for i in xrange(0,N):
			for j in xrange(0,N):
				mat[i*size:(i+1)*size, j*size:(j+1)*size] = obs_np[0,i*N+j,:,:].numpy()
		
		img = ax.imshow(mat, cmap='hot')
		fig.colorbar(img, ax=ax) 
		
		ax = fig.add_subplot(2, 1, 2, projection='3d')
		ax.plot(rx,ry,rz, '--', color='black', label = 'structure')
		for i, s in enumerate(self.sequence):
			if s=='A':
				ax.plot([rx[i]],[ry[i]],[rz[i]], '.r')
			if s=='B':
				ax.plot([rx[i]],[ry[i]],[rz[i]], '.b')

		ax.legend()
		plt.savefig("ABModelRender.png")
		plt.close(fig)
		



if __name__=='__main__':
	pass