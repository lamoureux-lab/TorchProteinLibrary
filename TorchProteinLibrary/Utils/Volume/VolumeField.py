import os
import sys
import torch
import enum

class ScalarField():
	
	def __init__(self, T, resolution=1.0, origin=[0,0,0]):
		# self.T = T.transpose(1,2).transpose(2,3).transpose(1,2).contiguous()
		self.T = T
		self.resolution = resolution
		self.origin = origin
		self.ftype = None
		
		if self.T.ndim == 4:
			self.T = self.T.squeeze()

		if self.T.ndim != 3:
			raise("Too many dimensions:", self.T.ndim())
		
		if self.T.is_cuda:
			self.T = self.T.cpu()
				
	def isosurface(self, isovalue, axis=None, edgecolor='k', facecolor='k', alpha=1.0):
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		from skimage import measure
		
		verts, faces, normals, values = measure.marching_cubes_lewiner(self.T.numpy(), isovalue, 
										spacing=(self.resolution, self.resolution, self.resolution))
		
		mesh = Poly3DCollection(verts[faces])
		mesh.set_edgecolor(edgecolor)
		mesh.set_facecolor(facecolor)
		mesh.set_alpha(alpha)
		
		fig = None
		if axis is None:
			fig = plt.figure(figsize=(10, 10))
			axis = fig.add_subplot(111, projection='3d')

		axis.set_xlabel("x")
		axis.set_ylabel("y")
		axis.set_zlabel("z")

		axis.set_xlim(0, self.T.size(0))
		axis.set_ylim(0, self.T.size(1))
		axis.set_zlim(0, self.T.size(2))
			
		axis.add_collection3d(mesh)
		if not fig is None:
			plt.tight_layout()
			plt.show()

if __name__=='__main__':
	
	box_size = 30
	res = 1.0
	scalar_field = torch.zeros(1, 30, 30, 30)
	# vector_field = torch.zeros(3, 30, 30, 30)
	for i in range(box_size):
		for j in range(box_size):
			for k in range(box_size):
				x = float(i)*res - float(box_size)*res/2.0
				y = float(j)*res - float(box_size)*res/2.0
				z = float(k)*res - float(box_size)*res/2.0
				scalar_field[0,i,j,k] = 1.0/(torch.sqrt(torch.tensor([x*x + y*y +z*z])).item()+1.0)
				x = float(i)*res - float(box_size)*res/2.0 + 5.0
				y = float(j)*res - float(box_size)*res/2.0
				z = float(k)*res - float(box_size)*res/2.0
				scalar_field[0,i,j,k] += 1.0/(torch.sqrt(torch.tensor([x*x + y*y +z*z])).item()+1.0)
				# d2 = scalar_field[0,i,j,k]*scalar_field[0,i,j,k]
				# vector_field[0,i,j,k] = torch.tensor([x])/(d2)
				# vector_field[1,i,j,k] = torch.tensor([y])/(d2)
				# vector_field[2,i,j,k] = torch.tensor([z])/(d2)
	
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(10, 10))
	axis = fig.add_subplot(111, projection='3d')
	ScalarField(scalar_field).isosurface(0.5, axis=axis, alpha=0.2)
	plt.plot([float(box_size)*res/2.0, float(box_size)*res/2.0-5], [float(box_size)*res/2.0, float(box_size)*res/2.0], [float(box_size)*res/2.0, float(box_size)*res/2.0], 'rx')
	plt.show()
	

	