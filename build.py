import os
import sys
import argparse

def make_cpp_lib(cpu_only=False, graham=False):
	if os.path.exists("build"):
		os.system("rm -r build")
	os.system("mkdir build")
	os.chdir('build')
	if graham:
		if cpu_only:
			os.system("cmake -DFAM_ONLY:BOOL=ON -DGRAHAM:BOOL=ON ..")
		else:
			os.system("cmake -DFAM_ONLY:BOOL=OFF -DGRAHAM:BOOL=ON ..")
	else:
		if cpu_only:
			os.system("cmake -DFAM_ONLY:BOOL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 ..")
		else:
			os.system("cmake -DFAM_ONLY:BOOL=OFF -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 ..")
			
	os.system("make")
	os.chdir('..')

def make_layer(dir_name, script_name):
	os.chdir(dir_name)
	os.system('python '+script_name)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Protein classes library build')	
	parser.add_argument('-cpu_only', default=False, help='If build only cpu')
	
	args = parser.parse_args()

	#Making cuda library
	make_cpp_lib(cpu_only = args.cpu_only, graham=False)
	
	cur_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
	
	#Building utilities
	pi_dir = os.path.join(cur_dir, 'PythonInterface')
	# make_layer(dir_name = pi_dir, script_name='build_layers.py')
	# make_layer(dir_name = pi_dir, script_name='build_utils.py')
	# make_layer(dir_name = pi_dir, script_name='build_visualization.py')

	#Building C_alpha_protein layers
	if not args.cpu_only:
		c_a_dir = os.path.join(pi_dir, 'C_alpha_protein')
		make_layer(dir_name = os.path.join(c_a_dir,'Angles2BasisDihedral'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Angles2CoordsAB'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Angles2CoordsDihedral'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Coords2Pairs'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Coords2RMSD'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Ddist2Forces'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Forces2DanglesAB'), script_name='build.py')
		make_layer(dir_name = os.path.join(c_a_dir,'Pairs2Distributions'), script_name='build.py')
	
	f_a_dir = os.path.join(pi_dir, 'FullAtomModel')
	make_layer(dir_name = os.path.join(f_a_dir,'Angles2Coords'), script_name='build.py')
	make_layer(dir_name = os.path.join(f_a_dir,'Coords2CenteredCoords'), script_name='build.py')
	make_layer(dir_name = os.path.join(f_a_dir,'Coords2TypedCoords'), script_name='build.py')
	make_layer(dir_name = os.path.join(f_a_dir,'TypedCoords2Volume'), script_name='build.py')
	make_layer(dir_name = os.path.join(f_a_dir,'Coords2RMSD'), script_name='build.py')
	make_layer(dir_name = os.path.join(f_a_dir,'PDB2Coords'), script_name='build.py')
	if not args.cpu_only:
		make_layer(dir_name = os.path.join(f_a_dir,'PDB2Volume'), script_name='build.py')
