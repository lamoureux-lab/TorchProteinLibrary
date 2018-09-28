import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
import matplotlib.pylab as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sea
import torch.optim as optim
from coords2RMSD import Coords2RMSD
def print_structure(loss, input_seq_len, i):
	# i = keys.index('2MLT_A')
	proteins, targets = loss.creator.get_aligned_coordinates()
	import matplotlib.pylab as plt
	import mpl_toolkits.mplot3d.axes3d as p3
	import seaborn as sea
	save_as_pdb(list(proteins[0].numpy().flatten()),'prot.pdb')
	save_as_pdb(list(targets[0].numpy().flatten()),'target.pdb')
	rx = proteins[i][:,0].numpy()
	ry = proteins[i][:,1].numpy()
	rz = proteins[i][:,2].numpy()

	tx = targets[i][:,0].numpy()
	ty = targets[i][:,1].numpy()
	tz = targets[i][:,2].numpy()

	print 'rmsd man=', np.sqrt(np.average( (rx-tx)*(rx-tx) + (ry-ty)*(ry-ty) + (rz-tz)*(rz-tz)))
	rmsd = 0.0
	for i in xrange(rx.shape[0]):
		rmsd += (rx[i]-tx[i])*(rx[i]-tx[i]) + (ry[i]-ty[i])*(ry[i]-ty[i]) + (rz[i]-tz[i])*(rz[i]-tz[i])
	rmsd = np.sqrt(rmsd/rx.shape[0])
	print 'rmds = ', rmsd
	
	max_x = np.max([np.max(tx),np.max(rx)])
	max_y = np.max([np.max(ty),np.max(ry)])
	max_z = np.max([np.max(tz),np.max(rz)])

	min_x = np.min([np.min(tx),np.min(rx)])
	min_y = np.min([np.min(ty),np.min(ry)])
	min_z = np.min([np.min(tz),np.min(rz)])
	min_ = np.min([min_x, min_y, min_z])
	max_ = np.max([max_x, max_y, max_z])

	fig = plt.figure()
	plt.title("Fitted C-alpha model and the protein C-alpha coordinates")
	ax = p3.Axes3D(fig)
	ax.plot(rx,ry,rz, '-ro', label = 'Prediction')
	ax.plot(tx,ty,tz, '-.bo', label = 'Target')
	ax.set_xlim(min_,max_)
	ax.set_ylim(min_,max_)
	ax.set_zlim(min_,max_)
	ax.legend()	
	plt.show()

def test():
	src = [0, 0, 0,
1.16665, 3.60399, 0.44514,
3.84597, 5.8857, 1.91587,
5.38486, 9.37098, 1.73532,
7.59245, 11.9271, 3.50749,
9.88289, 14.9268, 2.95644,
11.099, 18.2533, 4.37204,
14.0213, 20.7044, 4.3931,
15.0841, 24.331, 4.90889,
18.0202, 26.59, 5.81671,
19.4768, 30.1056, 5.55822,
21.8151, 32.6052, 7.2414,
24.1914, 35.5381, 6.69466,
25.6227, 38.6957, 8.28485,
28.5334, 41.1606, 8.29127,
29.5588, 44.7246, 9.18236,
32.3919, 46.9634, 10.411,
33.7769, 50.5167, 10.3527,
36.0937, 53.0155, 12.0663,
38.1124, 56.2145, 11.5772,
39.552, 59.3152, 13.2687,
42.1658, 62.0738, 12.9426,
43.1283, 65.6152, 13.9818,
46.0751, 67.8897, 14.8129,
47.3901, 71.4688, 14.7184,
49.7889, 73.924, 16.3814,]

	dst = [43.5938, -11.4844, 24.3281,
44.1875, -8.17969, 22.4688,
44, -10.2656, 19.3281,
40.375, -11.2578, 20.1094,
39.3438, -7.65625, 21.0781,
41.0625, -6.43359, 17.9531,
39.0938, -8.64844, 15.6328,
35.7812, -7.36719, 17.2188,
36.9062, -3.75977, 16.7188,
37.3438, -4.50781, 13.0859,
34.125, -6.40625, 12.3438,
31.5781, -6.59375, 15.1406,
31.9219, -2.91992, 16.3594,
31.0312, -1.27832, 13.0312,
27.8281, -3.41797, 12.7266,
26.9531, -2.64844, 16.3438,
27.3438, 1.05566, 15.6172,
24.8281, 0.827148, 12.7422,
22.4844, -1.08496, 15.0781,
22.625, 1.63281, 17.7656,
21.8906, 4.23438, 15.1562,
18.7344, 2.27344, 14.0234,
17.5, 1.92773, 17.5469,
18.0625, 5.67578, 18.2812,
15.8359, 6.73047, 15.3359,
13.1328, 4.35938, 16.5781,]
	L = len(src)/3
	maxL = L + 5
	x0 = Variable(torch.FloatTensor(src).cuda())
	x1 = Variable(torch.FloatTensor(dst).cuda())
	length = Variable(torch.IntTensor(1).fill_(L-1))
	loss = Coords2RMSD(maxL)
	rmsd_x0 = loss(x0, x1, length)
	print rmsd_x0

def test_pair_struct():
	src = [
		[1,0,0,
		2,0,0,
		3,0,0,
		4,0,0],
		[0,1,0,
		0,2,0,
		0,3,0,
		0,4,0]]

	dst = [
		[0,1,0,
		0,2,-0.1,
		0,3,0.1,
		0,4,0],
		[1,0,0,
		2,0,0,
		3,0,0,
		4,0,0]]

	L = len(src[0])/3
	maxL = L-1
	x0 = Variable(torch.FloatTensor(src).cuda())
	x1 = Variable(torch.FloatTensor(dst).cuda())
	print x0.size()
	length = Variable(torch.IntTensor(2).fill_(L-1))
	loss = Coords2RMSD(maxL)
	rmsd_x0 = loss(x0, x1, length)
	print rmsd_x0
	print_structure(rmsd_x0, length, 0)
	print_structure(rmsd_x0, length, 1)

def save_as_pdb(coords, filename):
	L = len(coords)/3
	with open(filename, 'w') as fout:
		for i in xrange(L):
			fout.write("ATOM    %3d  CA  LEU A   4    %8.3f%8.3f%8.3f \n"%(i, coords[3*i], coords[3*i+1],coords[3*i+2]))


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])


def test_pair_struct2():

	src = [ [0.0000,   0.0000,   0.0000,   1.1247,   3.6040,   0.5425,   3.6781,   5.8617,
2.2545,   5.5600,   9.1614,   1.9102,   7.2352,  12.0139,   3.8088,  10.1210,
14.5045,   3.6785,  11.2338,  18.0430,   4.5667,  14.2396,  20.1605,   5.5813,
15.7387,  23.6674,   5.5339,  17.9826,  26.0802,   7.4550,  20.6778,  28.7612,
7.1446,  22.1963,  31.8106,   8.8601,  25.3146,  33.9840 ,  9.1768 , 26.5117,
37.5519,   9.7974,  29.0874,  39.7191,  11.5909,  30.8998,  43.0715,  11.4336,
32.5543,  45.8463,  13.4613,  35.2976,  48.4948,  13.3762,  36.4438,  51.8771,
14.7157,  39.5079,  54.0554,  15.3590,  40.8319,  57.6213,  15.6404,  43.6385,
59.7395,  17.1185,  45.7953,  62.8561,  16.6911,  47.8349,  65.4654 , 18.5832,
50.5879,  68.0941,  18.3408,  51.9796,  71.2333,  20.0012,  ],[
0.0000,   0.0000,   0.0000 ,  1.1667 ,  3.6040 ,  0.4451 ,  3.8460  , 5.8857,
1.9159,   5.3849,   9.3710 ,  1.7353 ,  7.5925 , 11.9271  , 3.5075 ,  9.8829,
14.9268,   2.9564,  11.0990,  18.2533,   4.3720 , 14.0213 , 20.7044,   4.3931,
15.0841,  24.3310,   4.9089,  18.0202,  26.5900,   5.8167 , 19.4768 , 30.1056,
5.5582,  21.8151,  32.6052 ,  7.2414 , 24.1914  ,35.5381  , 6.6947 , 25.6227,
38.6957,   8.2849,  28.5334,  41.1606,   8.2913 , 29.5588 , 44.7246,   9.1824,
32.3919,  46.9634,  10.4110,  33.7769,  50.5167  ,10.3527 , 36.0937 , 53.0155,
12.0663,  38.1124,  56.2145,  11.5772 , 39.5520 , 59.3152 , 13.2687,  42.1658,
62.0738,  12.9426,  43.1283,  65.6152,  13.9818 , 46.0751 , 67.8897 , 14.8129,
47.3901,  71.4688,  14.7184,  49.7889,  73.9240 , 16.3814 ,  ]
]

	dst = [[
57.9375 , 34.7500 ,-22.8438 , 57.8438 , 38.0312 ,-24.8438 , 60.5938 , 40.6875,
-24.4531,  58.2500,  42.7812, -22.2812,  58.6562,  40.0938, -19.5625,  62.4062,
39.4688 ,-19.8594 , 63.1875 , 42.3438 ,-17.5312 , 60.9688 , 40.7188 ,-14.8672,
62.9375 , 37.5625 ,-15.4766 , 66.3750 , 39.1562 ,-14.9453 , 64.9375 , 40.9062,
-11.9219,  63.7500,  37.6250, -10.4922,  67.1250,  36.0000, -11.1172,  68.8750,
38.8750 , -9.3516 , 66.3750 , 38.6562 , -6.5234 , 67.2500 , 34.9688 , -6.0664,
70.9375 , 35.8438 , -5.8398 , 70.3750 , 38.5000 , -3.1758 , 68.1250 , 36.1875,
-1.1387 , 70.5625 , 33.2500 , -1.3857 , 73.2500 , 35.4688 ,  0.0320 , 70.7500,
36.6562 ,  2.6484 , 69.8750 , 33.0312 ,  3.3789 , 73.5625 , 32.2500 ,  3.8691,
73.6250 , 35.0625 ,  6.4609 , 71.3750 , 33.2500 ,  8.9844 
],[
43.5938, -11.4844 , 24.3281 , 44.1875 , -8.1797 , 22.4688 , 44.0000, -10.2656,
19.3281 , 40.3750 ,-11.2578 , 20.1094 , 39.3438 , -7.6562 , 21.0781,  41.0625,
-6.4336 , 17.9531 , 39.0938 , -8.6484 , 15.6328 , 35.7812 , -7.3672,  17.2188,
36.9062 , -3.7598 , 16.7188,  37.3438 , -4.5078 , 13.0859 , 34.1250,  -6.4062,
12.3438 , 31.5781 , -6.5938 , 15.1406 , 31.9219 , -2.9199 , 16.3594,  31.0312,
-1.2783 , 13.0312 , 27.8281 , -3.4180 , 12.7266 , 26.9531 , -2.6484,  16.3438,
27.3438 ,  1.0557 , 15.6172 , 24.8281  , 0.8271 , 12.7422 , 22.4844,  -1.0850,
15.0781 , 22.6250 ,  1.6328 , 17.7656 , 21.8906 ,  4.2344 , 15.1562,  18.7344,
2.2734  ,14.0234  ,17.5000  , 1.9277 , 17.5469 , 18.0625  , 5.6758 , 18.2812,
15.8359 ,  6.7305 , 15.3359 , 13.1328 ,  4.3594 , 16.5781   
]]
	# L = [len(src[0])/3-1, len(src[1])/3-1]
	save_as_pdb(src[1],'src.pdb')
	save_as_pdb(dst[1],'dst.pdb')
	L = [int(len(src[1])/3-1)]
	maxL = 27
	x0 = Variable(torch.FloatTensor(src[1]).cuda())
	x1 = Variable(torch.FloatTensor(dst[1]).cuda())
	print x0.size()
	length = Variable(torch.IntTensor(L))
	loss = Coords2RMSD(maxL)
	rmsd_x0 = loss(x0, x1, length)
	print torch.sqrt(rmsd_x0)
	print_structure(rmsd_x0, length, 0)
	# print_structure(rmsd_x0, length, 1)

	R = np.array([ 	[-914.291, 1494.97, 537.872, -7310.07],
				[1494.97, -6356.84, -3360.85, -1739.62],
				[537.872, -3360.85, 6714.98, -258.994],
				[-7310.07, -1739.62, -258.994, 556.146]])
	l, v = np.linalg.eig(R)
	q = v[:,np.argmax(l)]
	U = quaternion_matrix(q)
	r0 = np.array(src[1]).reshape((L[0]+1,3))
	t0 = np.array(dst[1]).reshape((L[0]+1,3))
	r0 -= np.average(r0, axis=0)
	t0 -= np.average(t0, axis=0)
	r1 = np.zeros((L[0]+1,3))
	for i in range(L[0]+1):
		r1[i,:] = U.dot(r0[i,:])
	
	fig = plt.figure()
	plt.title("Fitted C-alpha model and the protein C-alpha coordinates")
	ax = p3.Axes3D(fig)
	ax.plot(r1[:,0],r1[:,1],r1[:,2], '-ro', label = 'Prediction')
	ax.plot(t0[:,0],t0[:,1],t0[:,2], '-.bo', label = 'Target')
	plt.show()

	rmsd = 0.0
	rx, ry, rz = r1[:,0], r1[:,1], r1[:,2]
	tx, ty, tz = t0[:,0], t0[:,1], t0[:,2]
	for i in xrange(rx.shape[0]):
		rmsd += (rx[i]-tx[i])*(rx[i]-tx[i]) + (ry[i]-ty[i])*(ry[i]-ty[i]) + (rz[i]-tz[i])*(rz[i]-tz[i])
	rmsd = np.sqrt(rmsd/rx.shape[0])
	print 'rmds = ', rmsd

	print r1

if __name__=='__main__':
	# test()
	test_pair_struct2()