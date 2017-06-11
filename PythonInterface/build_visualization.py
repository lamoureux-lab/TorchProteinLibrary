import os
import torch
import glob
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)


here = os.path.normpath(os.path.dirname(__file__))
lib_dir = os.path.abspath(os.path.join(here, '../'))

sources = [ os.path.join(lib_dir,'VolumeVisualization/vol_viz_interface.cpp')]
headers = [ os.path.join(lib_dir,'VolumeVisualization/vol_viz_interface.h')]

include_dirs = [
	os.path.join(lib_dir, 'Math'),
    os.path.join(lib_dir, 'GL'),
    os.path.join(lib_dir, 'VolumeVisualization'),
]
library_dirs=[	os.path.join(lib_dir, 'build/GL'),
                os.path.join(lib_dir, 'build/VolumeVisualization'),
				os.path.join(lib_dir, 'build/Math'),
                    ]

defines = []
with_cuda = True

ffi = create_extension(
	'Visualization',
	headers=headers,
	sources=sources,
	define_macros=defines,
	relative_to=__file__,
	with_cuda=with_cuda,
	include_dirs = include_dirs,
	extra_compile_args=["-fopenmp"],
	extra_link_args = ["-lXxf86vm", "-lXext", "-lX11"],
    libraries = [ "GLFRAMEWORK", "MARCHINGCUBES", "TH_MATH", "GL", "GLU", "glut", "m"],
    library_dirs = library_dirs
)

if __name__ == '__main__':
	ffi.build()
	import Visualization
	print Visualization.__dict__.keys()