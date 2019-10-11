import os
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _Physics
import math
import numpy as np

def convertString(string):
	'''Converts a string to 0-terminated byte tensor'''
	return torch.from_numpy(np.fromstring(string, dtype=np.uint8))

class AtomNames2ParamsFunction(Function):
	@staticmethod
	def forward(ctx, resnames, atomnames, numatoms, types, params):
		ctx.save_for_backward(params)
		batch_size = resnames.size(0)
		max_num_atoms = resnames.size(1)
		assigned_params = torch.zeros(batch_size, max_num_atoms, 2, dtype=torch.double, device='cpu')

		ctx.dict = _Physics.AtomNames2Params_forward(resnames, atomnames, numatoms, types, params, assigned_params)		
		return assigned_params

	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		params, = ctx.saved_tensors
		gradInput = torch.zeros_like(params)
		
		_Physics.AtomNames2Params_backward(gradOutput, gradInput, ctx.dict)
		
		return None, None, None, None, gradInput

class AtomNames2Params(Module):
	def __init__(self):
		super(AtomNames2Params, self).__init__()
				
	def forward(self, resnames, atomnames, numatoms, types, params):
		return AtomNames2ParamsFunction.apply(resnames, atomnames, numatoms, types, params)

class ElectrostaticParameters:
	def __init__(self, path, type='amber'):
		charges_file = os.path.join(path, type+'.crg')
		radii_file = os.path.join(path, type+'.siz')
		charges = self.parseTinker(charges_file)
		radii = self.parseTinker(radii_file)

		common_keys = []
		for key in charges.keys():
			if key in radii.keys():
				common_keys.append(key)
		N = len(common_keys)
		# print("Num atom types:", N)
		
		self.types = torch.zeros(N, 2, 4, dtype=torch.uint8, device='cpu')
		self.params = torch.zeros(N, 2, dtype=torch.double, device='cpu')
		self.param_dict = {}

		for i, key in enumerate(common_keys):
			resname = "%-s"%key[0]
			atomname = "%-s"%key[1]
			self.types[i, 0, :len(resname)] = convertString(resname)[:len(resname)]
			self.types[i, 1, :len(atomname)] = convertString(atomname)[:len(atomname)]
			self.params[i, 0] = charges[key]
			self.params[i, 1] = radii[key]
			self.param_dict[key] = (charges[key], radii[key])
		
		
	def parseTinker(self, path):
		data = {}
		with open(path, 'r') as fin:
			for line in fin:
				sline = line[:line.find('!')].split()
				if len(sline)<2:
					continue
				if len(sline) == 2:
					atomname = sline[0]
					resname = ""
					param = float(sline[1])
				if len(sline) == 3:
					atomname = sline[0]
					resname = sline[1]
					param = float(sline[2])
								
				data[(resname, atomname)] = param

		return data
	

if __name__=='__main__':
	a2p = AtomNames2Params(ElectrostaticParameters('parameters'))
	
	a2p()