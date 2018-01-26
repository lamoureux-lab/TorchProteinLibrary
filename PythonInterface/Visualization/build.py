import os
import torch
import glob
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)


here = os.path.normpath(os.path.dirname(__file__))
lib_dir = os.path.abspath(os.path.join(here, '../../'))
sources = [ os.path.join(lib_dir,'Visualization/visualization_interface.cpp')
            ]
headers = [	os.path.join(lib_dir,'Visualization/visualization_interface.h')
			]

include_dirs = [
	os.path.join(lib_dir, 'Layers/FullAtomModel'),
    os.path.join(lib_dir, 'Visualization'),
    os.path.join(lib_dir, 'GL'),
	os.path.join(lib_dir, 'Math'),
]
library_dirs=[	os.path.join(lib_dir, 'build/Layers/FullAtomModel'),
                os.path.join(lib_dir, 'build/Visualization'),
                os.path.join(lib_dir, 'build/GL'),
				os.path.join(lib_dir, 'build/Math')]

defines = []
with_cuda = False

ffi = create_extension(
	'Exposed.cppVisualization',
	headers=headers,
	sources=sources,
	define_macros=defines,
	relative_to=__file__,
	with_cuda=with_cuda,
	include_dirs = include_dirs,
	extra_compile_args=["-fopenmp", "-std=c++11"],
	extra_link_args = [],
	libraries = ["FULL_ATOM_MODEL", "TH_MATH", "VISUALIZATION", "GLFRAMEWORK", "GL", "GLU", "glut"],
    library_dirs = library_dirs
)

if __name__ == '__main__':
	ffi.build()
	from Exposed import cppVisualization
	print cppVisualization.__dict__.keys()
