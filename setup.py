from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


if __name__=='__main__':
	FullAtomModel = CppExtension('_FullAtomModel', 
					sources = [
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/cVector3.cpp',
					'Layers/FullAtomModel/cConformation.cpp',
					'Layers/FullAtomModel/cConformationAA.cpp',
					'Layers/FullAtomModel/cGeometry.cpp',
					'Layers/FullAtomModel/cRigidGroup.cpp',
					'Layers/FullAtomModel/nUtil.cpp',
					'Layers/FullAtomModel/cPDBLoader.cpp',
					'Layers/FullAtomModel/Angles2Coords/angles2coords_interface.cpp',
					'Layers/FullAtomModel/PDB2Coords/pdb2coords_interface.cpp',
					'Layers/FullAtomModel/Coords2TypedCoords/coords2typedcoords_interface.cpp',
					'Layers/FullAtomModel/CoordsTransform/coordsTransform_interface.cpp',
					'Layers/FullAtomModel/main.cpp'],
					include_dirs = ['Layers/FullAtomModel',
					'Math'])
	Volume = CUDAExtension('_Volume',
					sources = [
					'Layers/Volume/TypedCoords2Volume/typedcoords2volume_interface.cpp',
					'Layers/Volume/Volume2Xplor/volume2xplor_interface.cpp',
					'Layers/Volume/Kernels.cu',
					'Layers/Volume/main.cpp'],
					include_dirs = ['Layers/Volume'])
	
	setup(name='TorchProteinLayers',
		ext_modules=[FullAtomModel, Volume],
		cmdclass={'build_ext': BuildExtension})
