import os
import sys

def make_cuda_lib():
	os.system("mkdir build")
	os.chdir('build')
	os.system("cmake -DFAM_ONLY:BOOL=OFF ..")
	os.system("make")
	os.chdir('..')

def make_layer(dir_name, script_name):
	os.chdir(dir_name)
	os.system('python '+script_name)

if __name__=='__main__':
	#Making cuda library
	make_cuda_lib()
	
	cur_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
	
	#Building utilities
	pi_dir = os.path.join(cur_dir, 'PythonInterface')
	make_layer(dir_name = pi_dir, script_name='build_layers.py')
	make_layer(dir_name = pi_dir, script_name='build_utils.py')
	make_layer(dir_name = pi_dir, script_name='build_visualization.py')

	#Building C_alpha_protein layers
	c_a_dir = os.path.join(pi_dir, 'C_alpha_protein')
	make_layer(dir_name = os.path.join(c_a_dir,'Angles2BasisDihedral'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Angles2CoordsAB'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Angles2CoordsDihedral'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Coords2Pairs'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Coords2RMSD'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Ddist2Forces'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Forces2DanglesAB'), script_name='build.py')
	make_layer(dir_name = os.path.join(c_a_dir,'Pairs2Distributions'), script_name='build.py')