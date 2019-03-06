import torch
from TorchProteinLibrary import ReducedModel
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3


if __name__=='__main__':
	a2b = ReducedModel.Angles2Backbone()
		
	#Setting conformation to alpha-helix
	N = 20
	num_aa = torch.zeros(1, dtype=torch.int, device='cuda').fill_(N)
	angles = torch.zeros(1, 3, N, dtype=torch.float, device='cuda')
	angles.data[:,0,:] = -1.047 # phi = -60 degrees
	angles.data[:,1,:] = -0.698 # psi = -40 degrees

	#Converting angles to coordinates
	coords = a2b(angles, num_aa)
	num_atoms = num_aa*3
	
	#Resizing coordinates array for convenience (to match selection mask)
	N = int(num_atoms.data[0])
	coords.resize_(1, N, 3)
	
	#Plotting backbone
	sx, sy, sz = coords[0,:,0].cpu().numpy(), coords[0,:,1].cpu().numpy(), coords[0,:,2].cpu().numpy()
	fig = plt.figure()
	plt.title("Reduced model")
	ax = p3.Axes3D(fig)
	ax.plot(sx,sy,sz, 'r-', label = 'backbone')
	ax.legend()
	ax.set_xlim(-0, 30)
	ax.set_ylim(-0, -30)
	ax.set_zlim(-0, 30)
	# plt.show()
	plt.savefig("ExampleAngles2Backbone.png")