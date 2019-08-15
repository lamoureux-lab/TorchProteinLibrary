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
				'TorchProteinLibrary.Volume.VolumeRotation',
				'TorchProteinLibrary.Volume.VolumeRMSD',
				#RMSD
				'TorchProteinLibrary.RMSD',
				'TorchProteinLibrary.RMSD.Coords2RMSD',
				]

	FullAtomModel = CUDAExtension('_FullAtomModel', 
					sources = [
					'Math/cVector3.cpp',
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/nUtil.cpp',
					'Layers/FullAtomModel/cConformation.cpp',
					'Layers/FullAtomModel/cConformationAA.cpp',
					'Layers/FullAtomModel/cGeometry.cpp',
					'Layers/FullAtomModel/cRigidGroup.cpp',
					'Layers/FullAtomModel/cPDBLoader.cpp',
					'Layers/FullAtomModel/Angles2Coords/angles2coords_interface.cpp',
					'Layers/FullAtomModel/PDB2Coords/pdb2coords_interface.cpp',
					'Layers/FullAtomModel/Coords2TypedCoords/coords2typedcoords_interface.cpp',
					'Layers/FullAtomModel/CoordsTransform/coordsTransform_interface.cpp',
					'Layers/FullAtomModel/CoordsTransform/coordsTransformGPU_interface.cpp',
					'Layers/FullAtomModel/TransformCUDAKernels.cu',
					'Layers/FullAtomModel/main.cpp'],
					include_dirs = ['Layers/FullAtomModel', 'Math'],
					libraries = ['gomp'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']})

	Volume = CUDAExtension('_Volume',
					sources = [
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/cVector3.cpp',
					'Math/nUtil.cpp',
					'Layers/Volume/TypedCoords2Volume/typedcoords2volume_interface.cpp',
					'Layers/Volume/Volume2Xplor/volume2xplor_interface.cpp',
					'Layers/Volume/Select/select_interface.cpp',
					'Layers/Volume/VolumeConvolution/volumeConvolution_interface.cpp',
					'Layers/Volume/VolumeRotation/volumeRotation_interface.cpp',
					'Layers/Volume/VolumeRMSD/volumeRMSD_interface.cpp',
					'Layers/Volume/Kernels.cu',
					'Layers/Volume/VolumeConv.cu',
					'Layers/Volume/RotateGrid.cu',
					'Layers/Volume/VolumeRMSD.cu',
					'Layers/Volume/main.cpp'],
					include_dirs = ['Layers/Volume', 'Math'],
					libraries = ['gomp', 'cufft'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']}
						)

	ReducedModel = CUDAExtension('_ReducedModel',
					sources = [
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/cVector3.cpp',
					'Math/nUtil.cpp',
					'Layers/ReducedModel/cBackboneProteinCPUKernels.cpp',
					'Layers/ReducedModel/Angles2Backbone/angles2backbone_interface.cpp',
					'Layers/ReducedModel/cBackboneProteinCUDAKernels.cu',
					'Layers/ReducedModel/main.cpp'],
					include_dirs = ['Layers/ReducedModel', 'Math'],
					libraries = ['gomp'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']}
					)

	RMSD = CppExtension('_RMSD',
					sources = [
					'Math/cMatrix33.cpp',
					'Math/cMatrix44.cpp',
					'Math/cVector3.cpp',
					'Math/nUtil.cpp',
					'Layers/RMSD/Coords2RMSD/coords2rmsd_interface.cpp',
					'Layers/RMSD/RMSDKernels.cu',
					'Layers/RMSD/main.cpp'],
					include_dirs = ['Layers/RMSD', 'Math'],
					libraries = ['gomp'],
					extra_compile_args={'cxx': ['-fopenmp'],
                                        'nvcc': ['-Xcompiler', '-fopenmp']})
	
	setup(	name='TorchProteinLibrary',
			version="0.1",
			ext_modules=[	RMSD,
							FullAtomModel, 
							Volume, 
							ReducedModel
						],
			cmdclass={'build_ext': BuildExtension},

			packages = Packages,
			author="Georgy Derevyanko",
			author_email="georgy.derevyanko@gmail.com",
			description="This is a collection of differentiable layers for pytorch, applicable to protein data",
			license="MIT",
			keywords="pytorch, protein structure, deep learning",
			url="https://github.com/lupoglaz/TorchProteinLibrary",
			project_urls={
				"Bug Tracker": "https://github.com/lupoglaz/TorchProteinLibrary/issues",
				"Documentation": "https://github.com/lupoglaz/TorchProteinLibrary/tree/Release/Doc",
				"Source Code": "https://github.com/lupoglaz/TorchProteinLibrary",
			})
