import os
import torch
import glob
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)


here = os.path.normpath(os.path.dirname(__file__))
lib_dir = os.path.abspath(os.path.join(here, '../'))

sources = [ os.path.join(lib_dir,'Utils/densityMap_interface.cpp')]
headers = [ os.path.join(lib_dir,'Utils/densityMap_interface.h')]

include_dirs = [
	os.path.join(lib_dir, 'Math'),
    os.path.join(lib_dir, 'Utils'),
]
library_dirs=[	
    os.path.join(lib_dir, 'build/Math'),
    os.path.join(lib_dir, 'build/Utils'),
]

defines = []
with_cuda = True

ffi = create_extension(
	'Utils',
	headers=headers,
	sources=sources,
	define_macros=defines,
	relative_to=__file__,
	with_cuda=with_cuda,
	include_dirs = include_dirs,
	extra_compile_args=["-fopenmp"],
	extra_link_args = [],
    libraries = ["TH_MATH", "UTILS"],
    library_dirs = library_dirs
)

if __name__ == '__main__':
	ffi.build()
	import Utils
	print Utils.__dict__.keys()