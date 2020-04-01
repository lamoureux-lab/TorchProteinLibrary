import os
import sys
import torch
import vtkplotter as vp
from vtkplotter import Line, Sphere

class BackboneStructure:
	def __init__(self):
		
		self.coords = None
		self.length = None
		self.atoms = []
		self.bonds = []
		self.at = 0

	def plot(self, coords, length, color=None, alpha=1.0):
		if coords.size(0) != 1:
			raise Exception("Plotting single protein", coords.size())
		if length.size(0) != 1:
			raise Exception("Plotting single protein", length.size())
		self.coords = coords
		self.length = length

		if len(self.atoms) == 0 and len(self.bonds) == 0:
			move = False
		else:
			move = True
			

		for i in range(self.length.item()):
			x = self.coords[0, 3*i + 0].item()
			y = self.coords[0, 3*i + 1].item()
			z = self.coords[0, 3*i + 2].item()
			if color is None:
				if i%3 == 0:
					_color = "blue"
				elif i%3 == 1:
					_color = "green"
				elif i%3 == 2:
					_color = "green"
			else:
				_color = color
				
			if move:
				self.atoms[i].pos([x,y,z])
			else:
				self.atoms.append(Sphere(pos=(x,y,z), r=0.5, c=_color, res=4, alpha=alpha))
			
			if i<(self.length.item()-1):
				xp1 = self.coords[0, 3*i + 3].item()
				yp1 = self.coords[0, 3*i + 4].item()
				zp1 = self.coords[0, 3*i + 5].item()
				if move:
					self.bonds[i].stretch(self.atoms[i].pos(), self.atoms[i+1].pos())
				else:
					self.bonds.append(Line(p0=[x,y,z], p1=[xp1,yp1,zp1], lw=5, c='black', alpha=alpha))
		return self.atoms, self.bonds

	def clear(self, plotter):
		for atom in self.atoms:
			plotter.remove(atom)
		self.atoms = []
		for bond in self.bonds:
			plotter.remove(bond)
		self.bonds = []

if __name__=='__main__':
	from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
	from ProteinStructure import ProteinStructure
	
	p2c = PDB2CoordsUnordered()
	prot = ProteinStructure(*p2c(["1brs.pdb"]))
	backbone = prot.select_CA()
	coords, num_atoms = backbone[0], backbone[-1]
	
	bb = BackboneStructure()
	import vtkplotter as vp
	v = vp.Plotter(title='basic shapes')
	atoms, bonds = bb.plot(coords, num_atoms)
	v.show(atoms, bonds, interactive=1)

