import numpy as np
import torch
import gym, time
from gym.spaces import Box
class TorchContinuous(Box):
	def __init__(self, low=-np.pi, high=np.pi, shape=None):
		super(TorchContinuous, self).__init__(low, high, shape)

	def sample(self):
		alpha = (np.max(self.high) - np.min(self.low))
		beta = np.min(self.low)
		return  ( torch.rand(self.shape).cuda() )*alpha + beta

	def __repr__(self):
		return "TorchContinuousTensor : "+str(self.shape)

	def to_jsonable(self, sample_n):
		raise(NotImplementedError)
	def from_jsonable(self, sample_n):
		raise(NotImplementedError)
	
if __name__=='__main__':
	space = TorchContinuous(low=-np.pi, high=np.pi, shape = (5,5))
	# print space
	sampl = space.sample()
	print sampl