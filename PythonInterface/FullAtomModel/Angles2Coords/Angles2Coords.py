import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
from Exposed import cppAngles2Coords
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from PDB2Coords import cppPDB2Coords
import numpy as np


def convertStringList(stringList):
    '''Converts list of strings to 0-terminated byte tensor'''
    maxlen = 0
    for string in stringList:
        string += '\0'
        if len(string)>maxlen:
            maxlen = len(string)
    ar = np.zeros( (len(stringList), maxlen), dtype=np.uint8)
    
    for i,string in enumerate(stringList):
        npstring = np.fromstring(string, dtype=np.uint8)
        ar[i,:npstring.shape[0]] = npstring
    
    return torch.from_numpy(ar)

def convertString(string):
    '''Converts a string to 0-terminated byte tensor'''  
    return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))


class Angles2CoordsFunction(Function):
	"""
	Protein angles -> coordinates function
	"""
		
	@staticmethod
	def forward(self, input_angles_cpu, sequenceTensor, num_atoms):
		# input_angles_cpu = input_angles_cpu.contiguous()
		
		max_num_atoms = torch.max(num_atoms)
		batch_size = input_angles_cpu.size(0)
		output_coords_cpu = torch.DoubleTensor(batch_size, 3*max_num_atoms).fill_(0.0).contiguous().zero_()
		output_resnames_cpu = torch.ByteTensor(batch_size, max_num_atoms, 4).contiguous().zero_()
		output_atomnames_cpu = torch.ByteTensor(batch_size, max_num_atoms, 4).contiguous().zero_()

		cppAngles2Coords.Angles2Coords_forward( sequenceTensor,
												input_angles_cpu,   #input angles
												output_coords_cpu,  #output coordinates
												output_resnames_cpu,
												output_atomnames_cpu,
												False
												)
		if math.isnan(output_coords_cpu.sum()):
			for i in xrange(batch_size):
				if math.isnan(output_coords_cpu[i,:].sum()):
					for j in xrange(self.num_atoms[i]):
						if math.isnan(output_coords_cpu[i, 3*j:3*j+3].sum()):
							print i, j, self.num_atoms[i], max_num_atoms
							print output_coords_cpu[i, 3*j-3:3*j]
							print output_coords_cpu[i, 3*j:3*j+3], output_atomnames_cpu[i, j, :].numpy().tostring(), output_resnames_cpu[i, j, :].numpy().tostring()
							break
			raise(Exception('Angles2CoordsFunction: forward Nan'))	
		
		self.save_for_backward(input_angles_cpu, sequenceTensor)

		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu
	
	@staticmethod	
	def backward(self, grad_atoms_cpu, *kwargs):
		# ATTENTION! It passes non-contiguous tensor
		grad_atoms_cpu = grad_atoms_cpu.contiguous()
		
		input_angles_cpu, sequenceTensor = self.saved_tensors
		
		batch_size = input_angles_cpu.size(0)
		grad_angles_cpu = torch.DoubleTensor(batch_size, input_angles_cpu.size(1), input_angles_cpu.size(2))
		grad_angles_cpu.fill_(0.0)
		
		cppAngles2Coords.Angles2Coords_backward(grad_atoms_cpu.data, grad_angles_cpu, sequenceTensor, input_angles_cpu, False)

		if math.isnan(grad_angles_cpu.sum()):
			raise(Exception('Angles2CoordsFunction: backward Nan'))		
		return Variable(grad_angles_cpu), None, None

class Angles2Coords(Module):
	def __init__(self, add_term=False):
		super(Angles2Coords, self).__init__()
		self.add_term = add_term
		self.num_atoms = None
		
	def forward(self, input_angles_cpu, sequences):
		stringListTensor = Variable(convertStringList(sequences))
				
		self.num_atoms = []
		for seq in sequences:
			self.num_atoms.append(cppPDB2Coords.getSeqNumAtoms(seq, self.add_term))
		num_atoms = Variable(torch.IntTensor(self.num_atoms))
		# print input_angles_cpu, stringListTensor, self.num_atoms

		return Angles2CoordsFunction.apply(input_angles_cpu, stringListTensor, num_atoms)