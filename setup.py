from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
import sysconfig

if __name__=='__main__':
	
	Packages = ['TorchProteinLibrary', 
				#FullAtomModel
				'TorchProteinLibrary.FullAtomModel', 
				'TorchProteinLibrary.FullAtomModel.Angles2Coords', 
				'TorchProteinLibrary.FullAtomModel.Coords2TypedCoords',
				'TorchProteinLibrary.FullAtomModel.CoordsTransform',
				'TorchProteinLibrary.FullAtomModel.PDB2Coords',
				#ReducedModel
				'TorchProteinLibrary.ReducedModel',
				'TorchProteinLibrary.ReducedModel.Angles2Backbone',
				#Volume
				'TorchProteinLibrary.Volume',
				'TorchProteinLibrary.Volume.TypedCoords2Volume',
				'TorchProteinLibrary.Volume.Select',
				'TorchProteinLibrary.Volume.VolumeConvolution',
				#RMSD
				'TorchProteinLibrary.RMSD',
				'TorchProteinLibrary.RMSD.Coords2RMSD',
				]

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
					include_dirs = ['Layers/FullAtomModel', 'Math'],
					libraries = ['gomp'],
					extra_compile_args=['-fopenmp'])
	Volume = CUDAExtension('_Volume',
					sources = [
					'Layers/Volume/TypedCoords2Volume/typedcoords2volume_interface.cpp',
					'Layers/Volume/Volume2Xplor/volume2xplor_interface.cpp',
					'Layers/Volume/Select/select_interface.cpp',
					'Layers/Volume/VolumeConvolution/volumeConvolution_interface.cpp',
					'Layers/Volume/Kernels.cu',
					'Layers/Volume/VolumeConv.cu',
					'Layers/Volume/main.cpp'],
					include_dirs = ['Layers/Volume'],
					libraries = ['gomp', 'cufft'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']}
						)

	ReducedModel = CUDAExtension('_ReducedModel',
					sources = [
					'Layers/ReducedModel/Angles2Backbone/angles2backbone_interface.cpp',
					'Layers/ReducedModel/cBackboneProteinCUDAKernels.cu',
					'Layers/ReducedModel/main.cpp'],
					include_dirs = ['Layers/ReducedModel'],
					libraries = ['gomp'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']}
					)

	RMSD_CPU = CppExtension('_RMSD_CPU',
					sources = [
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/cVector3.cpp',
					'Layers/RMSD/Coords2RMSD_CPU/coords2rmsd_interface.cpp',
					'Layers/RMSD/cRMSD.cpp',
					'Layers/RMSD/main_cpu.cpp'],
					include_dirs = ['Layers/RMSD', 'Math'])
	
	RMSD_GPU = CppExtension('_RMSD_GPU',
					sources = [
					'Layers/RMSD/Coords2RMSD_GPU/coords2rmsd_interface.cpp',
					'Layers/RMSD/RMSDKernels.cu',
					'Layers/RMSD/main_gpu.cpp'],
					include_dirs = ['Layers/RMSD'])
	
	setup(	name='TorchProteinLibrary',
			version="0.1",
			ext_modules=[	RMSD_CPU,
							RMSD_GPU,
							FullAtomModel, 
							Volume, 
							ReducedModel
						],
			cmdclass={'build_ext': BuildExtension},

			packages = Packages,
			author="Georgy Derevyanko",
			author_email="georgy.derevyanko@gmail.com",
			description="This is a collection of differentiable layers for pytorch, that applicable to protein data",
			license="MIT",
			keywords="pytorch, protein, deep learning",
			url="https://github.com/lupoglaz/TorchProteinLibrary",
			project_urls={
				"Bug Tracker": "https://github.com/lupoglaz/TorchProteinLibrary/issues",
				"Documentation": "https://github.com/lupoglaz/TorchProteinLibrary/tree/Release/Doc",
				"Source Code": "https://github.com/lupoglaz/TorchProteinLibrary",
			})
