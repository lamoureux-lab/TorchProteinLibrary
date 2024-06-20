#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  geometric_unit_coords.py
#  
#  Copyright 2020 Gabriele Orlando <orlando.gabriele89@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import torch

def write_pdb(coords, atom_list, resi_list,res_num,output_file):
	f=open(output_file,"w")
	coords=coords.data.numpy()
	for i in range(len(atom_list)):
		num=" "*(5-len(str(i)))+str(i)
		a_name=atom_list[i]+" "*(4-len(atom_list[i]))
		numres=" "*(4-len(str(res_num[i])))+str(res_num[i])
		
		x=round(coords[i][0],3)
		sx=str(x)
		while len(sx.split(".")[1])<3:
			sx+="0"
		x=" "*(8-len(sx))+sx
		
		y=round(coords[i][1],3)
		sy=str(y)
		while len(sy.split(".")[1])<3:
			sy+="0"
		y=" "*(8-len(sy))+sy
		
		z=round(coords[i][2],3)
		sz=str(z)
		while len(sz.split(".")[1])<3:
			sz+="0"
		z=" "*(8-len(sz))+sz
		
		f.write("ATOM  "+num+"  "+a_name+""+resi_list[i]+" A"+numres+"    "+x+y+z+"  1.00 64.10           "+atom_list[i][0]+"\n")
	f.close()
	
def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens,dev="cpu"): ### function taken from https://github.com/conradry/pytorch-rgn

	for i in range(3):
		#coordinates of last three atoms
		A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]

		#internal coordinates
		T = bond_angles[i]
		R = bond_lens[i]
		P = pred_torsions[:, i]
		
		#6x3 one triplet for each sample in the batch
		D2 = torch.stack([-R*torch.ones(P.size()).to(dev)*torch.cos(T), 
						  R*torch.cos(P)*torch.sin(T),
						  R*torch.sin(P)*torch.sin(T)], dim=1)

		#bsx3 one triplet for each sample in the batch
		BC = C - B
		bc = BC/torch.norm(BC, 2, dim=1, keepdim=True)

		AB = B - A

		N = torch.cross(AB, bc)
		n = N/torch.norm(N, 2, dim=1, keepdim=True)

		M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

		D = torch.bmm(M, D2.view(-1,3,1)).squeeze() + C
		pred_coords = torch.cat([pred_coords, D.view(1,-1,3)])
    
	return pred_coords

def main(angles):
	dev="cpu"
	batch_sz=1
	avg_bond_lens = torch.tensor([1.329, 1.459, 1.525]).to(dev)
	avg_bond_angles = torch.tensor([2.034, 2.119, 1.937]).to(dev)
	A = torch.tensor([0., 0., 1.]).to(dev)
	B = torch.tensor([0., 1., 0.]).to(dev)
	C = torch.tensor([1., 0., 0.]).to(dev)
	broadcast = torch.ones((batch_sz, 3)).to(dev)
	pred_coords = torch.stack([A*broadcast, B*broadcast, C*broadcast])
	for ix, triplet in enumerate(angles[:]):
		pred_coords = geometric_unit(pred_coords, triplet, 
									 avg_bond_angles, 
									 avg_bond_lens,dev=dev)
	
	pred_coords=pred_coords.transpose(0,1).squeeze(0)#.data.numpy()
	atom_list=["N","CA","C"]*(int(pred_coords.shape[0]/3))
	resi_list=['LYS', 'LYS', 'LYS', 'ASP', 'ASP', 'ASP', 'THR', 'THR', 'THR', 'THR', 'THR', 'THR', 'PHE', 'PHE', 'PHE', 'THR', 'THR', 'THR', 'LYS', 'LYS', 'LYS', 'ILE', 'ILE', 'ILE', 'PHE', 'PHE', 'PHE', 'VAL', 'VAL', 'VAL', 'GLY', 'GLY', 'GLY', 'GLY', 'GLY', 'GLY', 'LEU', 'LEU', 'LEU', 'PRO', 'PRO', 'PRO', 'TYR', 'TYR', 'TYR', 'HIS', 'HIS', 'HIS', 'THR', 'THR', 'THR', 'THR', 'THR', 'THR', 'ASP', 'ASP', 'ASP', 'ALA', 'ALA', 'ALA', 'SER', 'SER', 'SER', 'LEU', 'LEU', 'LEU'] 
	res_num=[]
	cont=0
	for i in range(pred_coords.shape[0]):
		if i%3==0:
			cont+=1
		res_num += [cont]
	write_pdb(pred_coords, atom_list, resi_list,res_num,output_file="test.pdb")
	
if __name__ == '__main__':
	## BB torsion angles taken from the pdb (psi,omega,phi)
	angles=  [[2.0393507,-3.08193687,-1.63904282],
			[1.88758931,3.13580149,-1.71118982],
			[0.35379004,3.08612295,-0.87478756],
			[-0.8281743,-3.01557253,-1.71838046],
			[0.08055055,3.1023487,-2.13224391],
			[-0.26218354,-3.13150822,-1.54592689],
			[2.01053773,-3.06474198,-1.81015267],
			[2.28223663,2.96027269,-1.4906815,],
			[2.26022762,-3.07716086,-2.11658479],
			[1.7943145,3.12770495,-1.81855047],
			[2.95888264,3.12862793,1.17557892],
			[0.42450582,3.10929074,-1.23104228],
			[2.56065955,2.9451381,-1.15394783],
			[2.61357919,-3.03015535,-1.17941288],
			[-0.22816098,3.09113235,-1.57717264],
			[-0.03619108,3.00468132,-1.12194906],
			[2.37864287,3.11036379,-2.00828385],
			[2.970595,2.9962374,-0.88871536],
			[-0.80831791,-3.05090645,-1.18386207],
			[-0.70809824,-3.09181413,-1.35472934],
			[-0.57338133,2.97493819,-1.03364934]]
	angles=torch.tensor(angles).unsqueeze(0).transpose(1,0)
	main(angles)
