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
from C_alpha_protein.Pairs2Distributions import cppPairs2Dist
from C_alpha_protein.Coords2RMSD import cppCoords2RMSD

class ABModelEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self):
		self.sequence = 'ABBABBABABBAB'
		self.num_bins = 10
		self.resolution = 0.3
		self.num_types = 2
		self.angles_length = len(self.sequence)-1
		self.num_atoms = len(self.sequence)
		self.beta = 0.1
		self.num_steps = 1000
		
		self.batch_size = 1
		self.angles = torch.squeeze(torch.FloatTensor(self.batch_size, 2, self.angles_length).cuda())
		self.A = torch.squeeze(torch.FloatTensor(self.batch_size, 16*self.angles_length).cuda())
		self.angles_length_tensor = torch.IntTensor(self.batch_size).fill_(self.angles_length)
		self.coords = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.new_coords = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.pairs = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms*self.num_atoms).cuda())
		
		self.rmsd = torch.FloatTensor(self.batch_size).cuda()
		self.c_coords_input = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.c_coords_target = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.U_coordinates_src = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.Ut_coordinates_dst = torch.squeeze(torch.FloatTensor(self.batch_size, 3*self.num_atoms).cuda())
		self.rot_mat_t = torch.squeeze(torch.FloatTensor(self.batch_size, 3, 3).cuda())

		self.mask = torch.ByteTensor(self.batch_size, self.num_atoms*self.num_atoms).cuda().fill_(0)
		self.mask[:, ::self.num_atoms+1]=1
		self.mask[:, 1::self.num_atoms+1]=1
		self.mask[:, self.num_atoms::self.num_atoms+1]=1
		self.mask.resize_(self.batch_size, self.num_atoms, self.num_atoms)
		self.mask = torch.squeeze(self.mask)
		
		self.curr_energy = torch.FloatTensor(self.batch_size).fill_(float('+inf')).cuda()
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
		
		self.action_space = TorchContinuous(low=-np.pi, high=np.pi, shape = self.angles.size())
		self.observation_space = TorchContinuous(low=-np.pi, high=np.pi, shape = self.angles.size())

		self.iterations = 0

		self.ones = torch.FloatTensor(self.batch_size).cuda().fill_(1.0)

		
	def compute_internal_variables(self, angles, coords):

		coords.fill_(0.0)
		cppAngles2CoordsAB.Angles2Coords_forward(   angles,              #input angles
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


	def compute_energy(self, angles):
		
		angles.resize_(self.batch_size, 2, self.angles_length)
		V1 = self.angles_length - torch.sum(torch.cos(angles[:,0,1:]), dim=1) - 1
		angles.squeeze_()
		
		self.pairs.resize_(self.batch_size, 3, self.num_atoms, self.num_atoms)
		r2 = torch.pow( self.pairs[:,0,:,:], 2) + torch.pow( self.pairs[:,1,:,:], 2) + torch.pow( self.pairs[:,2,:,:], 2)
		self.pairs.resize_(self.batch_size, 3*self.num_atoms*self.num_atoms).squeeze_()
		r6 = 1.0/torch.pow(r2, 3)
		r6.masked_fill_(self.mask, 0.0)
		r12 = torch.pow(r6, 2)
		r12 = torch.addcmul(r12, value=-1, tensor1=r6, tensor2=self.interaction)
		V2 = torch.sum(torch.sum(r12,1),1)
		
		return 0.25*V1+2.0*V2
	
	def _step(self, action):
				
		trial_angles = self.angles + action
		
		self.compute_internal_variables(trial_angles, self.new_coords)
		trial_energy = self.compute_energy(trial_angles)	
		
		prob = torch.min(self.ones, torch.exp(self.beta*(trial_energy - self.prev_energy)))
		samples = torch.rand(self.batch_size).cuda()
		real_acc = torch.le(prob, samples)
		
		if self.iterations>self.num_steps:
			done = True
		else:
			done = False

		self.iterations += 1

		if real_acc[0]:

			cppCoords2RMSD.Coords2RMSD_forward( self.coords, self.new_coords, self.rmsd, self.angles_length_tensor,
												self.c_coords_input,
												self.c_coords_target,
												self.U_coordinates_src,
												self.Ut_coordinates_dst,
												self.rot_mat_t)


			reward = 1.0 + self.rmsd[0]
			self.angles.copy_(trial_angles)
			self.prev_energy = trial_energy
			self.coords.copy_(self.new_coords)
						
		else:
			reward = 0.0
			

		return torch.FloatTensor(self.angles.size()).cuda().copy_(self.angles), reward, done, {}
		

		

	def _reset(self):
		self.angles.copy_(self.action_space.sample())
		self.iterations = 0
		self.compute_internal_variables(self.angles, self.coords)
		self.prev_energy = self.compute_energy(self.angles)
		return torch.FloatTensor(self.angles.size()).cuda().copy_(self.angles)

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
		
		
		fig = plt.figure()
		
		ax = fig.add_subplot(1, 1, 1, projection='3d')
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