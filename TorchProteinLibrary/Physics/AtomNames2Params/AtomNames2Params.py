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
		charges = self.unify_hydrogens(charges)

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

	def get_atom_hydrogens(self, amino_acid, atom_name):
		
		if atom_name == 'N' and (amino_acid != 'PRO'):
			return ['H']
			# return []
		if atom_name == 'OXT':
			return ['HXT']
		if (atom_name == 'CA') and (amino_acid != 'GLY'):
			return ['HA']
		if atom_name == 'C':
			return []
		if atom_name == 'O':
			return []
		
		if amino_acid == 'ALA': #http://ligand-expo.rcsb.org/reports/A/ALA/ALA_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB', '3HB']
		
		if amino_acid == 'ARG': #http://ligand-expo.rcsb.org/reports/A/ARG/ARG_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'CD':
				return ['1HD', '2HD']
			if atom_name == 'NE':
				return ['HE']
			if atom_name == 'CZ':
				return []
			if atom_name == 'NH1':
				return ['1HH1', '2HH1']
			if atom_name == 'NH2':
				return ['1HH2', '2HH2']
		
		if amino_acid == 'ASN': #http://ligand-expo.rcsb.org/reports/A/ASN/ASN_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'OD1':
				return []
			if atom_name == 'ND2':
				return ['1HD2', '2HD2']
		
		if amino_acid == 'ASP': #http://ligand-expo.rcsb.org/reports/A/ASP/ASP_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'OD1':
				return []
			if atom_name == 'OD2': #charged
				return []

		if amino_acid == 'CYS': #http://ligand-expo.rcsb.org/reports/C/CYS/CYS_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'SG':
				return ['HG']
		if amino_acid == 'CSS': #http://ligand-expo.rcsb.org/reports/C/CYS/CYS_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'SG':
				return []
		
		if amino_acid == 'GLU': #http://ligand-expo.rcsb.org/reports/G/GLU/GLU_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'CD':
				return []
			if atom_name == 'OE1':
				return []
			if atom_name == 'OE2': #charged
				return []
		
		if amino_acid == 'GLN': #http://ligand-expo.rcsb.org/reports/G/GLN/GLN_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'CD':
				return []
			if atom_name == 'OE1':
				return []
			if atom_name == 'NE2':
				return ['1HE2', '2HE2']

		if amino_acid == 'GLY': #http://ligand-expo.rcsb.org/reports/G/GLY/GLY_D3L3.gif
			if atom_name == 'CA':
				return ['1HA', '2HA']
		
		if amino_acid == 'HIS' or amino_acid == 'HIP': #http://ligand-expo.rcsb.org/reports/H/HIS/HIS_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'CD2':
				return ['HD2']
			if atom_name == 'NE2': #depends on the protonation state
				if amino_acid == 'HIS': return []
				if amino_acid == 'HIP': return ['HE2']
			if atom_name == 'CE1':
				return ['HE1']
			if atom_name == 'ND1':
				return ['HD1']

		if amino_acid == 'ILE': #http://ligand-expo.rcsb.org/reports/I/ILE/ILE_D3L3.gif
			if atom_name == 'CB':
				return ['HB']
			if atom_name == 'CG1':
				return ['1HG2', '2HG1']
			if atom_name == 'CD1':
				return ['1HD1', '2HD1', '3HD1']
			if atom_name == 'CG2':
				return ['1HG2', '2HG2', '3HG2']

		if amino_acid == 'LEU': #http://ligand-expo.rcsb.org/reports/L/LEU/LEU_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['HG']
			if atom_name == 'CD1':
				return ['1HD1', '2HD1', '3HD1']
			if atom_name == 'CD2':
				return ['1HD2', '2HD2', '3HD2']
		
		if amino_acid == 'LYS': #http://ligand-expo.rcsb.org/reports/L/LYS/LYS_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'CD':
				return ['1HD', '2HD']
			if atom_name == 'CE':
				return ['1HE', '2HE']
			if atom_name == 'NZ':
				return ['1HZ', '2HZ', '3HZ']
		
		if amino_acid == 'MET': #http://ligand-expo.rcsb.org/reports/M/MET/MET_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'SD':
				return []
			if atom_name == 'CE':
				return ['1HE', '2HE', '3HE']
		
		if amino_acid == 'PHE': #http://ligand-expo.rcsb.org/reports/P/PHE/PHE_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'CD1':
				return ['HD1']
			if atom_name == 'CD2':
				return ['HD2']
			if atom_name == 'CE1':
				return ['HE1']
			if atom_name == 'CE2':
				return ['HE2']
			if atom_name == 'CZ':
				return ['HZ']
		
		if amino_acid == 'PRO': #http://ligand-expo.rcsb.org/reports/P/PRO/PRO_D3L3.gif
			if atom_name == 'N':
				return []
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return ['1HG', '2HG']
			if atom_name == 'CD':
				return ['1HD', '2HD']
		
		if amino_acid == 'SER': #http://ligand-expo.rcsb.org/reports/S/SER/SER_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'OG':
				return ['HG']
		
		if amino_acid == 'THR': #http://ligand-expo.rcsb.org/reports/T/THR/THR_D3L3.gif
			if atom_name == 'CB':
				return ['HB']
			if atom_name == 'OG1':
				return ['HG1']
			if atom_name == 'CG2':
				return ['1HG2', '2HG2', '3HG2']
		
		if amino_acid == 'TRP': #http://ligand-expo.rcsb.org/reports/T/TRP/TRP_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'CD1':
				return ['HD1']
			if atom_name == 'CD2':
				return []
			if atom_name == 'NE1':
				return ['HE1']
			if atom_name == 'CE2':
				return []
			if atom_name == 'CZ2':
				return ['HZ2']
			if atom_name == 'CE3':
				return ['HE3']
			if atom_name == 'CZ3':
				return ['HZ3']
			if atom_name == 'CH2':
				return ['HH2']

		if amino_acid == 'TYR': #http://ligand-expo.rcsb.org/reports/T/TYR/TYR_D3L3.gif
			if atom_name == 'CB':
				return ['1HB', '2HB']
			if atom_name == 'CG':
				return []
			if atom_name == 'CD1':
				return ['HD1']
			if atom_name == 'CD2':
				return ['HD2']
			if atom_name == 'CE1':
				return ['HE1']
			if atom_name == 'CE2':
				return ['HE2']
			if atom_name == 'CZ':
				return []
			if atom_name == 'OH':
				return ['HH']

		if amino_acid == 'VAL': #http://ligand-expo.rcsb.org/reports/V/VAL/VAL_D3L3.gif
			if atom_name == 'CB':
				return ['HB']
			if atom_name == 'CG1':
				return ['1HG1', '2HG1', '3HG1']
			if atom_name == 'CG2':
				return ['1HG2', '2HG2', '3HG2']

		return []			
		# raise Exception('Unknown atom:', amino_acid, atom_name)
		

	def unify_hydrogens(self, charges):
		new_charges = {}
		for key in charges.keys():
			amino_acid = key[0]
			atom_name = key[1]

			if not (atom_name[0]=='C' or atom_name[0]=='O' or atom_name[0]=='N' or atom_name[0]=='S'):
				continue

			new_charges[key] = charges[key]
			hydrogens = self.get_atom_hydrogens(amino_acid, atom_name)

			for H_name in hydrogens:
				hydrogen_key = (amino_acid, H_name)
				if hydrogen_key in charges.keys():
					new_charges[key] += charges[hydrogen_key]
				else:
					print('Hydrogen not found:', hydrogen_key)

		return new_charges

		
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

	def saveTinker(self, path):
		print(self.param_dict)
		
		with open(path+'.crg', 'w') as fout:
			fout.write('atom__resnumbc_charge_\n')
			curr_res = None
			for res_name, atom_name in self.param_dict.keys():
				if curr_res is None:
					curr_res = res_name
				if curr_res != res_name:
					fout.write('\n')
					curr_res = res_name
				charge = self.param_dict[(res_name, atom_name)][0]
				fout.write('%-6s%-6s   %.4f\n'%(atom_name, res_name, charge))
		
		with open(path+'.siz', 'w') as fout:
			fout.write('atom__res_radius_\n')
			curr_res = None
			for res_name, atom_name in self.param_dict.keys():
				if curr_res is None:
					curr_res = res_name
				if curr_res != res_name:
					fout.write('\n')
					curr_res = res_name
				size = self.param_dict[(res_name, atom_name)][1]
				fout.write('%-6s%-6s%.4f\n'%(atom_name, res_name, size))

	

if __name__=='__main__':
	a2p = AtomNames2Params(ElectrostaticParameters('parameters'))
	
	a2p()