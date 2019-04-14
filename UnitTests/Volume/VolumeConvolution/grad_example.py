import sys
import os

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np
from numpy.fft import rfft, irfft
import seaborn as sea

def create_input(x0 = 0.0, size=300, resolution=0.1):
	y = np.zeros((size), dtype=float)
	half = int(size/2)
	for i in range(half):
		y[i] = np.exp( -(i*resolution - x0)*(i*resolution - x0))

	return y

def compute_corr(f, g, conj=True):
	f_star = rfft(f)
	g_star = rfft(g)
	if conj:
		g_star_conj = np.conj(g_star)
	else:
		g_star_conj = g_star
	corr_star = f_star * g_star_conj
	corr = irfft(corr_star)
	return np.real(corr)

def circ2norm(circ):
	size = circ.shape[0]
	half = int(size/2)
	norm = np.zeros((size,))
	norm[:half] = circ[half:]
	norm[half:] = circ[:half]
	return norm


def compute_grad(f, g, grad_output):
	grad_f = compute_corr(grad_output, g, conj=False)
	grad_g = compute_corr(f, grad_output, conj=True)
	return grad_f, grad_g

def compute_grad_num(f, g, func=None):
	size = f.shape[0]
	half = int(size/2)
	grad_f = np.zeros((size,))
	grad_g = np.zeros((size,))
	
	
	corr_init = compute_corr(f,g)
	E_init = func(corr_init)

	dx = 0.001
	for i in range(half):
		f_mod_p = np.copy(f) 
		f_mod_m = np.copy(f) 
		f_mod_p[i] += dx
		f_mod_m[i] -= dx
		corr_mod_p = compute_corr(f_mod_p, g)
		corr_mod_m = compute_corr(f_mod_m, g)
		E_mod_p = func(corr_mod_p)
		E_mod_m = func(corr_mod_m)
		grad_f[i] = (E_mod_p - E_mod_m)/(2*dx)

	for i in range(half):
		g_mod_p = np.copy(g) 
		g_mod_m = np.copy(g) 
		g_mod_p[i] += dx
		g_mod_m[i] -= dx
		corr_mod_p = compute_corr(f, g_mod_p)
		corr_mod_m = compute_corr(f, g_mod_m)
		E_mod_p = func(corr_mod_p)
		E_mod_m = func(corr_mod_m)
		grad_g[i] = (E_mod_p - E_mod_m)/(2*dx)

	return grad_f, grad_g


if __name__=='__main__':
	target = 12.5*create_input(x0=4.1)
	f = create_input(x0=7.0)
	g = create_input(x0=3.0)
	size = f.shape[0]
	half = int(size/2)
	
	corr_circ = compute_corr(f,g)
	corr = circ2norm(corr_circ)

	# grad_output = np.zeros((size,))
	# grad_output[250] = 1.0
	# grad_output[50] = -1.0
	
	# diff_output = corr_circ-target
	# zer = np.where(diff_output<1e-5, 0.0, diff_output)
	# grad_output = np.where(diff_output>0, 1.0, -1.0)*zer
	grad_output = 2.0*(corr_circ-target)

	def cost_func(corr):
		# return corr[250] - corr[50]
		# return np.sum(np.abs(corr-target))
		return np.sum((corr-target)*(corr-target))
	
	grad_input_f, grad_input_g = compute_grad(f, g, grad_output)
	num_grad_f, num_grad_g = compute_grad_num(f,g, cost_func)
	
	# plt.plot(f, 'g-', label = 'f')
	# plt.plot(g, 'g-', label = 'g')
	plt.plot(target, 'g-', label = 'target')
	plt.plot(corr_circ, 'y-', label = 'corr circular')
	plt.plot(grad_input_f, 'r-', label = 'analitical')
	plt.plot(num_grad_f[:half], 'bo', label = 'numerical')
	plt.legend()
	plt.savefig('corr_grad.png')

