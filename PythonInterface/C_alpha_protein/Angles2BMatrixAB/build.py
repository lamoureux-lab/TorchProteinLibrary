import os
import torch
import glob
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)


here = os.path.normpath(os.path.dirname(__file__))
lib_dir = os.path.abspath(os.path.join(here, '../../../'))
sources = [ os.path.join(lib_dir,'Layers/C_alpha_protein/Angles2BMatrix/angles2bmatrix_interface.cpp'),
			os.path.join(lib_dir,'Layers/C_alpha_protein/Angles2BMatrix/cAngles2BMatrix.cpp')]
headers = [	os.path.join(lib_dir,'Layers/C_alpha_protein/Angles2BMatrix/angles2bmatrix_interface.h')
			]

include_dirs = [
	os.path.join(lib_dir, 'Math'),
]
library_dirs=[	os.path.join(lib_dir, 'build/Layers/C_alpha_protein'),
				os.path.join(lib_dir, 'build/Math')]

defines = []
with_cuda = True

ffi = create_extension(
	'Exposed.cppAngles2BMatrixAB',
	headers=headers,
	sources=sources,
	define_macros=defines,
	relative_to=__file__,
	with_cuda=with_cuda,
	include_dirs = include_dirs,
	extra_compile_args=["-fopenmp"],
	extra_link_args = [],
	libraries = ["AB_MODEL", "TH_MATH"],
    library_dirs = library_dirs
)

if __name__ == '__main__':
	ffi.build()
	from Exposed import cppAngles2BMatrixAB
	print cppAngles2BMatrixAB.__dict__.keys()
