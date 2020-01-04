import os
import sys
import torch
import enum
import vtkplotter

class FieldType(enum.Enum):
	scalar = 1
	vector = 3

class VolumeField():
	
	def __init__(self, T, resolution=1.0, origin=[0,0,0]):
		self.T = T.transpose(1,2).transpose(2,3).transpose(1,2).contiguous()
		self.resolution = resolution
		self.origin = origin
		self.ftype = None
		
		if self.T.dim() != 4:
			raise("Too many dimensions:", self.T.ndim())
		
		if self.T.is_cuda:
			self.T = self.T.cpu()
		
		if self.T.size(0) == 1:
			self.ftype = FieldType.scalar
		elif self.T.size(0) == 3:
			self.ftype = FieldType.vector
		else:
			raise("Unknown field type:", self.T.size(0))

	def get_actor(self, **kwargs):
		if self.ftype == FieldType.scalar:
			return self.get_scalar_actor(kwargs)
		elif self.ftype == FieldType.vector:
			return self.get_vector_actor(kwargs)

	def get_scalar_actor(self, **kwargs):
		vol = vtkplotter.Volume(
					self.T[0,:,:,:].numpy(),
					origin=self.origin,
					spacing=[self.resolution, self.resolution, self.resolution])
		isos = vol.isosurface(**kwargs)	
		return isos
	
	def get_vector_actor(self, **kwargs):
		x = torch.linspace(self.origin[0], self.origin[0]+(self.T.size(1)-1)*self.resolution, self.T.size(1))
		y = torch.linspace(self.origin[1], self.origin[1]+(self.T.size(2)-1)*self.resolution, self.T.size(2))
		z = torch.linspace(self.origin[2], self.origin[2]+(self.T.size(3)-1)*self.resolution, self.T.size(3))
		
		grid_x, grid_y, grid_z = torch.meshgrid(x,y,z)
		sources = torch.stack([grid_x, grid_y, grid_z], dim=3)
		sources = sources.view(self.T.size(1)*self.T.size(2)*self.T.size(3), 3).numpy()
		deltas = self.T.transpose(0,3).contiguous()
		deltas = deltas.view(self.T.size(1)*self.T.size(2)*self.T.size(3), 3).numpy()
		
		return vtkplotter.Arrows(sources, sources + deltas, **kwargs)

if __name__=='__main__':
	
	box_size = 30
	res = 1.0
	scalar_field = torch.zeros(1, 30, 30, 30)
	vector_field = torch.zeros(3, 30, 30, 30)
	for i in range(box_size):
		for j in range(box_size):
			for k in range(box_size):
				x = float(i)*res - float(box_size)*res/2.0
				y = float(j)*res - float(box_size)*res/2.0
				z = float(k)*res - float(box_size)*res/2.0
				scalar_field[0,i,j,k] = torch.sqrt(torch.tensor([x*x + y*y +z*z]))
				d2 = scalar_field[0,i,j,k]*scalar_field[0,i,j,k]
				vector_field[0,i,j,k] = torch.tensor([x])/(d2)
				vector_field[1,i,j,k] = torch.tensor([y])/(d2)
				vector_field[2,i,j,k] = torch.tensor([z])/(d2)
	
	import vtkplotter as vp
	scalar_field = VolumeField(scalar_field)
	vector_field = VolumeField(vector_field)
	vector = vector_field.get_actor()
	scalar = scalar_field.get_actor(threshold=10.0)
	

	vp = vtkplotter.Plotter(N=2, title='basic shapes', axes=0)
	vp.sharecam = True
	vp.show(scalar, at=0)
	vp.show(vector, at=1, interactive=1)
	

	