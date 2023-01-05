import torch
from torch import optim
import numpy as np
from Bio.PDB import calc_angle, rotaxis, Vector, calc_dihedral
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Polypeptide import d3_to_index, dindex_to_1

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def _tensor2str(tensor):
	return (tensor.numpy().astype(dtype=np.uint8).tostring().split(b'\00')[0]).decode("utf-8")

def Coords2BioStructure(coords, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type):
	batch_size = coords.size(0)
	structures = []
	length = torch.zeros(batch_size, dtype=torch.int, device='cpu')
	for batch_idx in range(batch_size):
		struct = Structure(" ")
		model = Model(0)
		
		previous_resnum = resnums[batch_idx, 0].item()
		
		atom_idx = 0 
		chain_name = _tensor2str(chainnames[batch_idx, atom_idx, :])
		previous_chain = chain_name
		current_chain = Chain(chain_name)
		
		residue_3name = _tensor2str(resnames[batch_idx, atom_idx, :])
		# residue_1name = dindex_to_1[d3_to_index[residue_3name]]
		current_residue = Residue((" ", resnums[batch_idx, atom_idx].item(), " "), residue_3name, current_chain.get_id())
		
		for atom_idx in range(num_atoms[batch_idx].item()):
			if previous_resnum < resnums[batch_idx, atom_idx].item():
				residue_3name = _tensor2str(resnames[batch_idx, atom_idx, :])
				# residue_1name = dindex_to_1[d3_to_index[residue_3name]]
				
				current_chain.add(current_residue)
				current_residue = Residue((" ", resnums[batch_idx, atom_idx].item(), " "), residue_3name, current_chain.get_id())
				previous_resnum = resnums[batch_idx, atom_idx].item()
			
			if previous_chain != _tensor2str(chainnames[batch_idx, atom_idx, :]):
				chain_name = _tensor2str(chainnames[batch_idx, atom_idx, :])
				previous_chain = chain_name
				model.add(current_chain)
				current_chain = Chain(chain_name)

			atom_name = _tensor2str(atomnames[batch_idx, atom_idx, :])
			coord = coords[batch_idx, 3*atom_idx:3*atom_idx+3].numpy()

			if polymer_type == 0:
				atom = Atom(atom_name, coord, 0.0, 1.0, "", atom_name, None)
				current_residue.add(atom)
			if polymer_type == 1:
				#error from pdb.atom [for poly type: change how atom is saved and then save as pdb to be read by barnaba]
				# print(atom_name, coord, 0.0, 1.0, "", atom_name, None)
				atom = Atom(atom_name, coord, 0.0, 1.0, "", atom_name, None)
				current_residue.add(atom)

			if polymer_type == 2:
				atom = Atom(atom_name, coord, 0.0, 1.0, "", atom_name, None)
				current_residue.add(atom)

			
		current_chain.add(current_residue)
		model.add(current_chain)
		struct.add(model)
		structures.append(struct)
		length[batch_idx] = len(list(struct.get_residues()))

	return structures, length

def BioStructure2Dihedrals(structure, polymer_type):
	if polymer_type == 0:
		residues = list(structure.get_residues())
		angles = torch.zeros(8, len(residues), dtype=torch.double, device='cpu')
		phi, psi, omega = getBackbone(residues, polymer_type)
		for i, residue in enumerate(residues):
			angles[0, i] = phi[i]
			angles[1, i] = psi[i]
			angles[2, i] = omega[i]
			xis = getRotamer(residue)
			for j, xi in enumerate(xis):
				angles[3+j, i] = xis[j]
		return angles
	if polymer_type == 1:
		residues = list(structure.get_residues())
		angles = torch.zeros(24, len(residues), dtype=torch.double, device='cpu')
		alpha, beta, gamma, delta, epsilon, zeta, nu0, nu1, nu2, nu3, nu4, chi = getBackbone(residues, polymer_type)
		for i, residue in enumerate(residues):
			angles[0, i] = alpha[i]
			angles[1, i] = beta[i]
			angles[2, i] = gamma[i]
			angles[3, i] = delta[i]
			angles[4, i] = epsilon[i]
			angles[5, i] = zeta[i]
			angles[6, i] = nu0[i]
			angles[7, i] = nu1[i]
			angles[8, i] = nu2[i]
			angles[9, i] = nu3[i]
			angles[10, i] = nu4[i]
			angles[11, i] = chi[i]
			# print(str(residue)[10])
			xis = getRotamer(residue, 1)
			for j, xi in enumerate(xis):
				angles[12 + j, i] = xis[j]
		return angles
		# print("Error Polymer Type 1 Not Implemented for TPL/TPL/FullAtomModel/Coords2PDB.py/Biostructure2Dihedrals")

def getBackbone(residues, polymer_type= 0):
	if polymer_type == 0:
		phi = [0.0]
		psi = []
		omega = []
		for i, res_i in enumerate(residues):
			N_i = res_i["N"].get_vector()
			CA_i = res_i["CA"].get_vector()
			C_i = res_i["C"].get_vector()
			if i>0:
				res_im1 = residues[i-1]
				C_im1 = res_im1["C"].get_vector()
				phi.append(calc_dihedral(C_im1, N_i, CA_i, C_i))
			if i<(len(residues)-1):
				res_ip1 = residues[i+1]
				N_ip1 = res_ip1["N"].get_vector()
				psi.append(calc_dihedral(N_i, CA_i, C_i, N_ip1))

			if i<(len(residues)-1):
				res_ip1 = residues[i+1]
				N_ip1 = res_ip1["N"].get_vector()
				CA_ip1 = res_ip1["CA"].get_vector()
				omega.append(calc_dihedral(CA_i, C_i, N_ip1, CA_ip1))

		psi.append(0.0)
		omega.append(0.0)
		return phi, psi, omega

	if polymer_type == 1:
		alpha = []
		beta = []
		gamma = []
		delta = []
		epsilon = []
		zeta = []
		nu0 = []
		nu1 = []
		nu2 = []
		nu3 = []
		nu4 = []
		chi = []

		chain_idx = str(0)

		for i, res_i in enumerate(residues):
			# print(res_i.get_list())
			# print(res_i.get_parent())
			if str(res_i.get_parent()) > chain_idx:  #& res_i.get_atom() == "O5'":
				chain_idx = str(res_i.get_parent())
				#res_idx = res_i

				#if res_i == res_idx:
				O5_i = res_i["O5'"].get_vector()
				C5_i = res_i["C5'"].get_vector()
				C4p_i = res_i["C4'"].get_vector()
				C3_i = res_i["C3'"].get_vector()
				O3_i = res_i["O3'"].get_vector()
				C1_i = res_i["C1'"].get_vector()
				C2_i = res_i["C2'"].get_vector()
				O4_i = res_i["O4'"].get_vector()

				if str(res_i)[10] == "A" or str(res_i)[10] == "G":
					N_i = res_i["N9"].get_vector()
					C_i = res_i["C4"].get_vector()

				if str(res_i)[10] == "C" or str(res_i)[10] == "T":
					N_i = res_i["N1"].get_vector()
					C_i = res_i["C2"].get_vector()

				alpha.append(0.0)
				beta.append(0.0)

				gamma.append(calc_dihedral(O5_i, C5_i, C4p_i, C3_i))

				delta.append(calc_dihedral(C5_i, C4p_i, C3_i, O3_i))

				if i < (len(residues) - 1):
					res_ip1 = residues[i + 1]
					P_ip1 = res_ip1["P"].get_vector()
					epsilon.append(calc_dihedral(C4p_i, C3_i, O3_i, P_ip1))

				if i < (len(residues) - 1):
					res_ip1 = residues[i + 1]
					P_ip1 = res_ip1["P"].get_vector()
					O5_ip1 = res_ip1["O5'"].get_vector()
					zeta.append(calc_dihedral(C3_i, O3_i, P_ip1, O5_ip1))

				nu0.append(calc_dihedral(C4p_i, O4_i, C1_i, C2_i))
				nu1.append(calc_dihedral(O4_i, C1_i, C2_i, C3_i))
				nu2.append(calc_dihedral(C1_i, C2_i, C3_i, C4p_i))
				nu3.append(calc_dihedral(C2_i, C3_i, C4p_i, O4_i))
				nu4.append(calc_dihedral(C3_i, C4p_i, O4_i, C1_i))
				chi.append(calc_dihedral(O4_i, C1_i, N_i, C_i))

				continue



			# print(res_i)
			P_i = res_i["P"].get_vector()
			O5_i = res_i["O5'"].get_vector()
			C5_i = res_i["C5'"].get_vector()
			C4p_i = res_i["C4'"].get_vector()
			C3_i = res_i["C3'"].get_vector()
			O3_i = res_i["O3'"].get_vector()
			C1_i = res_i["C1'"].get_vector()
			C2_i = res_i["C2'"].get_vector()
			O4_i = res_i["O4'"].get_vector()


			if str(res_i)[10] == "A" or str(res_i)[10] == "G":
				N_i = res_i["N9"].get_vector()
				C_i = res_i["C4"].get_vector()

			if str(res_i)[10] == "C" or str(res_i)[10] == "T":
				N_i = res_i["N1"].get_vector()
				C_i = res_i["C2"].get_vector()

			if i > 0:
				res_im1 = residues[i - 1]
				O3_im1 = res_im1["O3'"].get_vector()
				alpha.append(calc_dihedral(O3_im1, P_i, O5_i, C5_i))


			beta.append(calc_dihedral(P_i, O5_i, C5_i, C4p_i))

			gamma.append(calc_dihedral(O5_i, C5_i, C4p_i, C3_i))

			delta.append(calc_dihedral(C5_i, C4p_i, C3_i, O3_i))



			if i <(len(residues) - 1):
				res_ip1 = residues[i + 1]
				if str(res_ip1.get_parent()) > chain_idx:
					epsilon.append(0.0)
					zeta.append(0.0)
					nu0.append(calc_dihedral(C4p_i, O4_i, C1_i, C2_i))
					nu1.append(calc_dihedral(O4_i, C1_i, C2_i, C3_i))
					nu2.append(calc_dihedral(C1_i, C2_i, C3_i, C4p_i))
					nu3.append(calc_dihedral(C2_i, C3_i, C4p_i, O4_i))
					nu4.append(calc_dihedral(C3_i, C4p_i, O4_i, C1_i))
					chi.append(calc_dihedral(O4_i, C1_i, N_i, C_i))
					continue

			if i < (len(residues) - 1):
				res_ip1 = residues[i + 1]
				P_ip1 = res_ip1["P"].get_vector()
				epsilon.append(calc_dihedral(C4p_i, C3_i, O3_i, P_ip1))

			if i < (len(residues) - 1):
				res_ip1 = residues[i + 1]
				P_ip1 = res_ip1["P"].get_vector()
				O5_ip1 = res_ip1["O5'"].get_vector()
				zeta.append(calc_dihedral(C3_i, O3_i, P_ip1, O5_ip1))

			if i == (len(residues) - 1):
				epsilon.append(0.0)
				zeta.append(0.0)

			nu0.append(calc_dihedral(C4p_i, O4_i, C1_i, C2_i))
			nu1.append(calc_dihedral(O4_i, C1_i, C2_i, C3_i))
			nu2.append(calc_dihedral(C1_i, C2_i, C3_i, C4p_i))
			nu3.append(calc_dihedral(C2_i, C3_i, C4p_i, O4_i))
			nu4.append(calc_dihedral(C3_i, C4p_i, O4_i, C1_i))
			chi.append(calc_dihedral(O4_i, C1_i, N_i, C_i))

			# print(chi)

		return alpha, beta, gamma, delta, epsilon, zeta, nu0, nu1, nu2, nu3, nu4, chi

def getRotamer(residue, polymer_type = 0):
	if polymer_type == 0:

		if residue.get_resname()=='GLY':
			return []
		if residue.get_resname()=='ALA':
			return []
		if residue.get_resname()=='CYS':
			return getCysRot(residue)
		if residue.get_resname()=='SER':
			return getSerRot(residue)
		if residue.get_resname()=='VAL':
			return getValRot(residue)
		if residue.get_resname()=='ILE':
			return getIleRot(residue)
		if residue.get_resname()=='LEU':
			return getLeuRot(residue)
		if residue.get_resname()=='THR':
			return getThrRot(residue)
		if residue.get_resname()=='ARG':
			return getArgRot(residue)
		if residue.get_resname()=='LYS':
			return getLysRot(residue)
		if residue.get_resname()=='ASP':
			return getAspRot(residue)
		if residue.get_resname()=='ASN':
			return getAsnRot(residue)
		if residue.get_resname()=='GLU':
			return getGluRot(residue)
		if residue.get_resname()=='GLN':
			return getGlnRot(residue)
		if residue.get_resname()=='MET':
			return getMetRot(residue)
		if residue.get_resname()=='HIS':
			return getHisRot(residue)
		if residue.get_resname()=='PRO':
			return getProRot(residue)
		if residue.get_resname()=='PHE':
			return getPheRot(residue)
		if residue.get_resname()=='TYR':
			return getTyrRot(residue)
		if residue.get_resname()=='TRP':
			return getTrpRot(residue)

	if polymer_type == 1:
		print(str(residue)[10])
		if str(residue)[10] == 'A':
			return getDARot(residue)
		if str(residue)[10] == 'C':
			return getDCRot(residue)
		if str(residue)[10] == 'G':
			return getDGRot(residue)
		if str(residue)[10] == 'T':
			return getDTRot(residue)

def getCysRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	SG = residue["SG"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, SG) + np.pi
	return [xi1]

def getSerRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	OG = residue["OG"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, OG) + np.pi
	return [xi1]

def getValRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG1 = residue["CG1"].get_vector()
	CG2 = residue["CG2"].get_vector()
	xi1 = (calc_dihedral(N, CA, CB, CG1) + calc_dihedral(N, CA, CB, CG2))/2.0 + np.pi
	return [xi1]

def getIleRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG1 = residue["CG1"].get_vector()
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG1)
	xi2 = calc_dihedral(CA, CB, CG1, CD1)+np.pi
	return [xi1, xi2]

def getLeuRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()
	CD1 = residue["CD1"].get_vector()
	CD2 = residue["CD2"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = (calc_dihedral(CA, CB, CG, CD1) + calc_dihedral(CA, CB, CG, CD2))/2.0
	return [xi1, xi2]

def getThrRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	OG1 = residue["OG1"].get_vector()	
	CG2 = residue["CG2"].get_vector()
	xi1 = (calc_dihedral(N, CA, CB, OG1) + calc_dihedral(N, CA, CB, CG2))/2.0
	return [xi1]

def getArgRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	NE = residue["NE"].get_vector()
	CZ = residue["CZ"].get_vector()
	NH1 = residue["NH1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, NE)
	xi4 = calc_dihedral(CG, CD, NE, CZ)
	xi5 = calc_dihedral(CD, NE, CZ, NH1)
	return [xi1, xi2, xi3, xi4, xi5]

def getLysRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	CE = residue["CE"].get_vector()
	NZ = residue["NZ"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, CE)
	xi4 = calc_dihedral(CG, CD, CE, NZ)
	return [xi1, xi2, xi3, xi4]

def getAspRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	OD1 = residue["OD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, OD1)
	return [xi1, xi2]

def getAsnRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	OD1 = residue["OD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, OD1)
	return [xi1, xi2]

def getGluRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	OE1 = residue["OE1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, OE1)
	return [xi1, xi2, xi3]

def getGlnRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD = residue["CD"].get_vector()
	OE1 = residue["OE1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD)
	xi3 = calc_dihedral(CB, CG, CD, OE1)
	return [xi1, xi2, xi3]

def getMetRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	SD = residue["SD"].get_vector()
	CE = residue["CE"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, SD)
	xi3 = calc_dihedral(CB, CG, SD, CE)
	return [xi1, xi2, xi3]

def getHisRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	ND1 = residue["ND1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, ND1)+np.pi/2.0
	return [xi1, xi2]

def getProRot(residue):
	return []

def getPheRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)+np.pi/2.0
	return [xi1, xi2]

def getTyrRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)+np.pi/2.0
	return [xi1, xi2]

def getTrpRot(residue):
	N = residue["N"].get_vector()
	CA = residue["CA"].get_vector()
	CB = residue["CB"].get_vector()
	CG = residue["CG"].get_vector()	
	CD1 = residue["CD1"].get_vector()
	xi1 = calc_dihedral(N, CA, CB, CG)
	xi2 = calc_dihedral(CA, CB, CG, CD1)-np.pi/2.0
	return [xi1, xi2]

def getDARot(residue):
	C1 = residue["C1'"].get_vector()
	N9 = residue["N9"].get_vector()
	C4 = residue["C4"].get_vector()
	N3 = residue["N3"].get_vector()
	C2 = residue["C2"].get_vector()
	N1 = residue["N1"].get_vector()
	C6 = residue["C6"].get_vector()
	N6 = residue["N6"].get_vector()
	C5 = residue["C5"].get_vector()
	N7 = residue["N7"].get_vector()
	C8 = residue["C8"].get_vector()
	xi1 = calc_dihedral(C1, N9, C4, N3)
	xi2 = calc_dihedral(N9, C4, N3, C2)
	xi3 = calc_dihedral(C4, N3, C2, N1)
	xi4 = calc_dihedral(N3, C2, N1, C6)
	xi5 = calc_dihedral(C2, N1, C6, C5)
	xi6 = calc_dihedral(C2, N1, C6, N6)
	xi7 = calc_dihedral(N1, C6, C5, C4) #or N7
	xi8 = calc_dihedral(C6, C5, C4, N9) #or N3
	xi9 = calc_dihedral(C6, C5, N7, C8)
	xi10 = calc_dihedral(C5, N7, C8, N9)
	xi11 = calc_dihedral(N7, C8, N9, C1) #or C4

	return [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10, xi11]

def getDCRot(residue):
	C1 = residue["C1'"].get_vector()
	N1 = residue["N1"].get_vector()
	C2 = residue["C2"].get_vector()
	O2 = residue["O2"].get_vector()
	N3 = residue["N3"].get_vector()
	C4 = residue["C4"].get_vector()
	N4 = residue["N4"].get_vector()
	C5 = residue["C5"].get_vector()
	C6 = residue["C6"].get_vector()
	xi1 = calc_dihedral(C1, N1, C2, N3)
	xi2 = calc_dihedral(C1, N1, C2, O2)
	xi3 = calc_dihedral(N1, C2, N3, C4)
	xi4 = calc_dihedral(C2, N3, C4, C5)
	xi5 = calc_dihedral(C2, N3, C4, N4)
	xi6 = calc_dihedral(N3, C4, C5, C6)
	xi7 = calc_dihedral(C4, C5, C6, N1)
	xi8 = calc_dihedral(C5,C6, N1, C1)

	return [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8]

def getDGRot(residue):
	C1 = residue["C1'"].get_vector()
	N9 = residue["N9"].get_vector()
	C4 = residue["C4"].get_vector()
	N3 = residue["N3"].get_vector()
	C2 = residue["C2"].get_vector()
	N2 = residue["N2"].get_vector()
	N1 = residue["N1"].get_vector()
	C6 = residue["C6"].get_vector()
	O6 = residue["O6"].get_vector()
	C5 = residue["C5"].get_vector()
	N7 = residue["N7"].get_vector()
	C8 = residue["C8"].get_vector()
	xi1 = calc_dihedral(C1, N9, C4, N3)
	xi2 = calc_dihedral(N9, C4, N3, C2)
	xi3 = calc_dihedral(C4, N3, C2, N1)
	xi4 = calc_dihedral(C4, N3, C2, N2)
	xi5 = calc_dihedral(N3, C2, N1, C6)
	xi6 = calc_dihedral(C2, N1, C6, C5)
	xi7 = calc_dihedral(C2, N1, C6, O6)
	xi8 = calc_dihedral(N1, C6, C5, C4)  # or N7
	xi9 = calc_dihedral(C6, C5, C4, N9)  # or N3
	xi10 = calc_dihedral(C6, C5, N7, C8)
	xi11 = calc_dihedral(C5, N7, C8, N9)
	xi12 = calc_dihedral(N7, C8, N9, C1)  # or C4

	return [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10, xi11, xi12]

def getDTRot(residue):
	C1 = residue["C1'"].get_vector()
	N1 = residue["N1"].get_vector()
	C2 = residue["C2"].get_vector()
	O2 = residue["O2"].get_vector()
	N3 = residue["N3"].get_vector()
	C4 = residue["C4"].get_vector()
	O4 = residue["O4"].get_vector()
	C5 = residue["C5"].get_vector()
	C6 = residue["C6"].get_vector()
	C7 = residue["C7"].get_vector()
	xi1 = calc_dihedral(C1, N1, C2, N3)
	xi2 = calc_dihedral(C1, N1, C2, O2)
	xi3 = calc_dihedral(N1, C2, N3, C4)
	xi4 = calc_dihedral(C2, N3, C4, C5)
	xi5 = calc_dihedral(C2, N3, C4, O4)
	xi6 = calc_dihedral(N3, C4, C5, C6)
	xi7 = calc_dihedral(N3, C4, C5, C7)
	xi8 = calc_dihedral(C4, C5, C6, N1)
	xi9 = calc_dihedral(C5, C6, N1, C1)

	return [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9]

def Coords2Angles(coords, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type=0):
	if polymer_type == 0:

		structures, length = Coords2BioStructure(coords, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type)
		max_seq_length = max(length)
		batch_size = length.size(0)
		angles = torch.zeros(batch_size, 8, max_seq_length, dtype=torch.float32, device='cpu')
		for batch_idx, structure in enumerate(structures):
			dihedrals = BioStructure2Dihedrals(structure, polymer_type)
			angles[batch_idx,:,:length[batch_idx].item()] = dihedrals

	if polymer_type == 1:
		structures, length = Coords2BioStructure(coords, chainnames, resnames, resnums, atomnames, num_atoms, polymer_type)
		# print("length", length)
		max_seq_length = max(length)
		batch_size = length.size(0)
		angles = torch.zeros(batch_size, 12, max_seq_length, dtype=torch.float32, device='cpu')
		for batch_idx, structure in enumerate(structures):
			dihedrals = BioStructure2Dihedrals(structure, polymer_type)
			angles[batch_idx, :, :length[batch_idx].item()] = dihedrals
		print(angles)
		print("Error Polymer Type 1 Not Implemented in TPL/TPL/FullAtomModel/Coords2Angles.py")

	elif polymer_type == 2:
		length = 0
		angles = 0
		print("Error Polymer Type 2 Not Implemented in TPL/TPL/FullAtomModel/Coords2Angles.py")

	else:
		print("Polymer Type is Not Valid")

	return angles, length


if __name__=='__main__':
	from TorchProteinLibrary.FullAtomModel import Angles2Coords
	a2c = Angles2Coords()
	sequences = ["AAA", "AAA"]
	angles = torch.zeros(2, 8, len(sequences[0]), dtype=torch.float, device='cpu')
	coords, chainnames, resnames, resnums, atomnames, num_atoms = a2c(angles, sequences)
	angles, length = Coords2Angles(coords, chainnames, resnames, resnums, atomnames, num_atoms)
	print(length)
	print(angles)
