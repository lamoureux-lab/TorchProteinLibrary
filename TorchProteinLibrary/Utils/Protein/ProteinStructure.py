import os
import sys
import torch

class ProteinStructure:
	def __init__(self, coords, chains, resnames, resnums, atomnames, numatoms):
		self.set(coords, chains, resnames, resnums, atomnames, numatoms)

	def set(self, coords, chains, resnames, resnums, atomnames, numatoms):
		self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = coords, \
		chains, resnames, resnums, atomnames, numatoms

	def get(self):
		return self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms

	def select_atoms_mask(self, atomic_mask):
		N = self.numatoms[0].item()
		
		isSel = atomic_mask
		isSel_coords = torch.stack([atomic_mask for i in range(3)], dim=1).unsqueeze(dim=0)
		isSel_names = torch.stack([atomic_mask for i in range(4)], dim=1).unsqueeze(dim=0)	
		num_sel_atoms =  atomic_mask.sum().item()
		sel_numatoms = torch.tensor([num_sel_atoms], dtype=torch.int, device='cpu')	
		
		coords = self.coords.view(1, N, 3)
		sel_coords = torch.masked_select(coords, isSel_coords).view(1, num_sel_atoms*3).contiguous()
		sel_chains = torch.masked_select(self.chains, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnames = torch.masked_select(self.resnames, isSel_names).view(1, num_sel_atoms, 4).contiguous()
		sel_resnums = torch.masked_select(self.resnums, isSel).view(1, num_sel_atoms).contiguous()
		sel_atomnames = torch.masked_select(self.atomnames, isSel_names).view(1, num_sel_atoms, 4).contiguous()

		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms

		# self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = sel_coords, \
		# sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms 

	def select_CA(self):
		is0C = torch.eq(self.atomnames[:,:,0], 67).squeeze()
		is1A = torch.eq(self.atomnames[:,:,1], 65).squeeze()
		is20 = torch.eq(self.atomnames[:,:,2], 0).squeeze()
		isCA = is0C*is1A*is20

		return self.select_atoms_mask(isCA)

	def select_chain(self, chain_name):
		is0C = torch.eq(self.chains[:,:,0], ord(chain_name)).squeeze()
		is10 = torch.eq(self.chains[:,:,1], 0).squeeze()
		isChain = is0C*is10

		return self.select_atoms_mask(isChain)

	def select_residues_list(self, atom_list):
		N = self.numatoms[0].item()
		
		coords = self.coords.view(1, N, 3)
		sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames = [], [], [], [], []
		sel_numatoms = 0
		for chain, resnum, resname in atom_list:
			for i in range(N):
				if (chain == str(chr(self.chains[0, i, 0].item()))) and (resnum == self.resnums[0, i].item()):
					sel_coords.append(coords[:, i, :])
					sel_chains.append(self.chains[:, i, :])
					sel_resnames.append(self.resnames[:, i, :])
					sel_resnums.append(self.resnums[:, i])
					sel_atomnames.append(self.atomnames[:, i, :])
					sel_numatoms += 1
		
		sel_coords = torch.stack(sel_coords, dim=1).view(1, sel_numatoms*3).contiguous()
		sel_chains = torch.stack(sel_chains, dim=1).contiguous()
		sel_resnames = torch.stack(sel_resnames, dim=1).contiguous()
		sel_resnums = torch.stack(sel_resnums, dim=1).contiguous()
		sel_atomnames = torch.stack(sel_atomnames, dim=1).contiguous()
		sel_numatoms = torch.tensor([sel_numatoms], dtype=torch.int, device='cpu').contiguous()

		return sel_coords, sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms
		# self.coords, self.chains, self.resnames, self.resnums, self.atomnames, self.numatoms = sel_coords, \
		# sel_chains, sel_resnames, sel_resnums, sel_atomnames, sel_numatoms

	def plot_coords(self, axis = None, type='line', args = {}):
		import matplotlib 
		import matplotlib.pylab as plt
		import mpl_toolkits.mplot3d.axes3d as p3
		
		if axis is None:
			fig = plt.figure()
			axis = p3.Axes3D(fig)
		
		N = self.numatoms[0].item()
		chains = set([])
		for i in range(N):
			chains.add(str(chr(self.chains[0,i,0])))

		for chain in chains:
			prot_chain = self.select_chain(chain)
			coords = prot_chain[0].view(1, prot_chain[-1].item(), 3)
			sx, sy, sz = coords[0,:,0].numpy(), coords[0,:,1].numpy(), coords[0,:,2].numpy()
			if type=='line':
				axis.plot(sx, sy, sz, label = chain, **args)
			elif type=='scatter':
				axis.scatter(sx, sy, sz, label = chain, **args)
			else:
				raise(Exception("Unknown plot type", type))
		
		coords = self.coords.view(1, N, 3)
		x, y, z = coords[0,:,0], coords[0,:,1], coords[0,:,2]
		
		ax_min_x, ax_max_x = axis.get_xlim()
		ax_min_y, ax_max_y = axis.get_ylim()
		ax_min_z, ax_max_z = axis.get_zlim()

		#Preserving aspect ratio
		min_x = min(torch.min(x).item(), ax_min_x)
		max_x = max(torch.max(x).item(), ax_max_x)
		min_y = min(torch.min(y).item(), ax_min_y)
		max_y = max(torch.max(y).item(), ax_max_y)
		min_z = min(torch.min(z).item(), ax_min_z)
		max_z = max(torch.max(z).item(), ax_max_z)
		max_L = max([max_x - min_x, max_y - min_y, max_z - min_z])
		axis.set_xlim(min_x, min_x+max_L)
		axis.set_ylim(min_y, min_y+max_L)
		axis.set_zlim(min_z, min_z+max_L)
		
		if not fig is None:
			axis.legend()
			plt.show()

	def plot_tube(self):
		import vtk
		colors = vtk.vtkNamedColors()
		colors.SetColor("BkgColor", 0.9, 0.9, 0.9, 1.0)

		N = self.numatoms[0].item()
		points = vtk.vtkPoints()
		points.SetNumberOfPoints(N)
		for i in range(N):
			points.SetPoint(i, self.coords[0,3*i+0].item(), self.coords[0,3*i+1].item(), self.coords[0,3*i+2].item())

		polyLine = vtk.vtkPolyLine()
		polyLine.GetPointIds().SetNumberOfIds(N)
		for i in range(N):
			polyLine.GetPointIds().SetId(i, i)
		
		# print(polyLine)
		
		cells = vtk.vtkCellArray()
		cells.InsertNextCell(polyLine)
		polyData = vtk.vtkPolyData()
		polyData.SetPoints(points)
		polyData.SetLines(cells)

		tubeFilter = vtk.vtkTubeFilter()
		tubeFilter.SetInputData(polyData)
		tubeFilter.SetRadius(1.0)
		tubeFilter.SetNumberOfSides(16)

		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(tubeFilter.GetOutputPort())
		
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
		actor.GetProperty().SetLineWidth(5)
		
		return actor

	def plot_atoms(self, colors=None):
		import vtk
		
		N = self.numatoms[0].item()
		points = vtk.vtkPoints()
		points.SetNumberOfPoints(N)
		for i in range(N):
			points.SetPoint(i, self.coords[0,3*i+0].item(), self.coords[0,3*i+1].item(), self.coords[0,3*i+2].item())

		bonds = vtk.vtkCellArray()
		this_coords = self.coords.view(N, 3)
		sep_mat = this_coords.unsqueeze(dim=1) - this_coords.unsqueeze(dim=0)
		dist2_mat = (sep_mat*sep_mat).sum(dim=2)

		for i in range(N):
			for j in range(N):
				if i >= j: continue
				if dist2_mat[i,j].item() < 3.0:
					bond = vtk.vtkLine()
					bond.GetPointIds().SetId(0, i)
					bond.GetPointIds().SetId(1, j)
					bonds.InsertNextCell(bond)
		
		atom_colors = vtk.vtkUnsignedCharArray()
		atom_colors.SetName('Colors')
		atom_colors.SetNumberOfComponents(3)
		atoms = vtk.vtkCellArray()
		
		for i in range(N):
			vertex = vtk.vtkVertex()
			vertex.GetPointIds().SetId(0, i)
			atoms.InsertNextCell(vertex)
			if colors is None:
				if self.atomnames[0,i,0].item() == ord('C'):
					atom_colors.InsertNextTuple3(100, 200, 100)
				elif self.atomnames[0,i,0].item() == ord('N'):
					atom_colors.InsertNextTuple3(100, 100, 200)
				elif self.atomnames[0,i,0].item() == ord('O'):
					atom_colors.InsertNextTuple3(200, 100, 100)
				else:
					atom_colors.InsertNextTuple3(200, 200, 200)
			else:
				atom_colors.InsertNextTuple3(colors[i,0].item(), colors[i,1].item(), colors[i,2].item())
			
		polyData = vtk.vtkPolyData()
		polyData.SetPoints(points)
		polyData.SetVerts(atoms)
		polyData.SetLines(bonds)
		polyData.GetPointData().SetScalars(atom_colors)
		# polyData.AddArray(atom_colors)
		
		# Bonds actor
		tubeFilter = vtk.vtkTubeFilter()
		tubeFilter.SetInputData(polyData)
		tubeFilter.SetRadius(0.3)
		tubeFilter.SetNumberOfSides(3)	
				
		mapper_bonds = vtk.vtkPolyDataMapper()
		mapper_bonds.SetInputConnection(tubeFilter.GetOutputPort())
		
		actor_bonds = vtk.vtkLODActor()
		actor_bonds.SetMapper(mapper_bonds)
		actor_bonds.GetProperty().SetRepresentationToSurface()
		actor_bonds.GetProperty().SetInterpolationToGouraud()
		actor_bonds.GetProperty().SetAmbient(0.15)
		actor_bonds.GetProperty().SetDiffuse(0.85)
		actor_bonds.GetProperty().SetSpecular(0.1)
		actor_bonds.GetProperty().SetSpecularPower(30)
		actor_bonds.GetProperty().SetSpecularColor(1, 1, 1)
		actor_bonds.GetProperty().SetDiffuseColor(1.0000, 0.8941, 0.70981)
		
		# Atoms actor
		sphere = vtk.vtkSphereSource()
		sphere.SetCenter(0,0,0)
		sphere.SetRadius(2.0)

		glyph = vtk.vtkGlyph3D()
		glyph.SetInputData(polyData)
		glyph.SetOrient(1)
		glyph.SetColorMode(1)
		glyph.SetScaleMode(2)
		glyph.SetScaleFactor(0.25)
		glyph.SetSourceConnection(sphere.GetOutputPort())

		mapper_atoms = vtk.vtkPolyDataMapper()
		mapper_atoms.SetInputConnection(glyph.GetOutputPort())
				
		actor_atoms = vtk.vtkLODActor()
		actor_atoms.SetMapper(mapper_atoms)
		actor_atoms.GetProperty().SetRepresentationToSurface()
		actor_atoms.GetProperty().SetInterpolationToGouraud()
		actor_atoms.GetProperty().SetAmbient(0.15)
		actor_atoms.GetProperty().SetDiffuse(0.85)
		actor_atoms.GetProperty().SetSpecular(0.1)
		actor_atoms.GetProperty().SetSpecularPower(30)
		actor_atoms.GetProperty().SetSpecularColor(1, 1, 1)
		actor_atoms.SetNumberOfCloudPoints(30000)
		
		return actor_bonds, actor_atoms


def unite_proteins(receptor, ligand):
	lcoords, lchains, lres_names, lres_nums, latom_names, lnum_atoms = ligand.get()
	rcoords, rchains, rres_names, rres_nums, ratom_names, rnum_atoms = receptor.get()
	ccoords = torch.cat([lcoords, rcoords], dim=1).contiguous()
	cchains = torch.cat([lchains, rchains], dim=1).contiguous()
	cres_names = torch.cat([lres_names, rres_names], dim=1).contiguous()
	cres_nums = torch.cat([lres_nums, rres_nums], dim=1).contiguous()
	catom_names = torch.cat([latom_names, ratom_names], dim=1).contiguous()
	cnum_atoms = lnum_atoms + rnum_atoms

	complex = ProteinStructure(ccoords, cchains, cres_names, cres_nums, catom_names, cnum_atoms)
	return complex


if __name__=='__main__':
	from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
	
	p2c = PDB2CoordsUnordered()
	prot = ProteinStructure(*p2c(["1brs.pdb"]))
	atoms_plot = prot.plot_atoms()
	prot = ProteinStructure(*prot.select_CA())
	backbone_plot = prot.plot_tube()

	import vtkplotter as vp
	v = vp.Plotter(N=2, title='basic shapes', axes=0)
	v.sharecam = True
	v.show(backbone_plot, at=0)
	v.show(atoms_plot, at=1, interactive=1)

