import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from celluloid import Camera

from TorchProteinLibrary.Volume import VolumeCrossMultiply

def gaussian(tensor, center=(0,0,0), sigma=1.0):
	center = center[0] + tensor.size(0)/2, center[1] + tensor.size(1)/2, center[2] + tensor.size(2)/2
	for x in range(tensor.size(0)):
		for y in range(tensor.size(1)):
			for z in range(tensor.size(2)):
				r2 = (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) + (z-center[2])*(z-center[2])
				arg = torch.tensor([-r2/sigma])
				tensor[x,y,z] = torch.exp(arg)

def test_translations(a, b, dim=0):
	mult = VolumeCrossMultiply()

	fig = plt.figure(figsize=(10, 10))
	cam = Camera(fig)
	ax = fig.add_subplot(111, projection='3d')

	for i in range(20):
		R = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0)
		T = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0)
		T[0,1] = float(i)-10
		R[0,2] = 2*np.pi*float(i)/20.0
		v = mult(b.unsqueeze(dim=0).unsqueeze(dim=1), a.unsqueeze(dim=0).unsqueeze(dim=1),R,T)
		
		verts, faces, normals, values = measure.marching_cubes_lewiner(mult.transformed_volume.squeeze().numpy(), 0.01)

		mesh = Poly3DCollection(verts[faces])
		mesh.set_edgecolor('k')
		ax.add_collection3d(mesh)

		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		ax.set_xlim(0, 20)
		ax.set_ylim(0, 20)
		ax.set_zlim(0, 20)

		plt.tight_layout()
		cam.snap()
	
	animation = cam.animate()
	plt.show()

def test_rotations(a, b, dim):
	mult = VolumeCrossMultiply()

	fig = plt.figure(figsize=(10, 10))
	cam = Camera(fig)
	ax = fig.add_subplot(111, projection='3d')

	for i in range(20):
		R = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0)
		T = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0)
		R[0,dim] = 2*np.pi*float(i)/20.0
		v = mult(b.unsqueeze(dim=0).unsqueeze(dim=1), a.unsqueeze(dim=0).unsqueeze(dim=1), R, T)
		
		verts, faces, normals, values = measure.marching_cubes_lewiner(mult.transformed_volume.squeeze().numpy(), 0.01)

		mesh = Poly3DCollection(verts[faces])
		mesh.set_edgecolor('k')
		ax.add_collection3d(mesh)

		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		ax.set_xlim(0, 20)
		ax.set_ylim(0, 20)
		ax.set_zlim(0, 20)

		plt.tight_layout()
		cam.snap()
	
	animation = cam.animate()
	plt.show()

def test_optimization(a,b):
	from torch import optim
	mult = VolumeCrossMultiply()
	R = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0).requires_grad_()
	T = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(dim=0).requires_grad_()
	
	fig = plt.figure(figsize=(10, 10))
	cam = Camera(fig)
	ax = fig.add_subplot(211, projection='3d')
	ax1 = fig.add_subplot(212)

	optimizer = optim.Adam([R, T], lr = 0.1)
	L = []
	for i in range(100):
		optimizer.zero_grad()
		m = mult(a.unsqueeze(dim=0).unsqueeze(dim=1), b.unsqueeze(dim=0).unsqueeze(dim=1), R, T)
		loss = -m.sum()
		L.append(loss.item())
		loss.backward()
		optimizer.step()
		print(mult.transformed_volume.min(), mult.transformed_volume.max())
		bverts, bfaces, bnormals, bvalues = measure.marching_cubes_lewiner(torch.abs(mult.transformed_volume).detach().squeeze().numpy(), 0.1)
		averts, afaces, anormals, avalues = measure.marching_cubes_lewiner(torch.abs(a).detach().squeeze().numpy(), 0.1)

		bmesh = Poly3DCollection(bverts[bfaces])
		bmesh.set_edgecolor('k')
		amesh = Poly3DCollection(averts[afaces])
		amesh.set_edgecolor('r')
		ax.add_collection3d(bmesh)
		ax.add_collection3d(amesh)

		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		ax.set_xlim(0, 30)
		ax.set_ylim(0, 30)
		ax.set_zlim(0, 30)

		plt.tight_layout()
		cam.snap()
	
	ax1.plot(L)
	animation = cam.animate()
	animation.save('animation.mp4')
	# plt.show()

if __name__=='__main__':
	a = torch.zeros(20,20,20)
	b = torch.zeros(20,20,20)
	t = torch.zeros(20,20,20)
	gaussian(a, (-2,0,0), 1.0)
	gaussian(b, (2,0,0), 1.0)
	a += b

	# test_rotations(a, b, dim=2)
	test_translations(a, b, dim=2)
	sys.exit()
	a = torch.zeros(30,30,30)
	b = torch.zeros(30,30,30)
	t = torch.zeros(30,30,30)
	gaussian(a, (0,0,0), 3.0)
	gaussian(t, (4.9,0,0), 3.0)
	a-=t
	gaussian(b, (5,0,0), 3.0)    
	gaussian(t, (5,5.1,0), 3.0)    
	b-=t

	test_optimization(a,b)
	