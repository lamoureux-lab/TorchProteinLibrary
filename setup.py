from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


if __name__=='__main__':
	PytorchInterace = [	#FullAtomModel
						'TorchProteinLibrary/__init__.py',
						'TorchProteinLibrary/FullAtomModel/__init__.py',
						'TorchProteinLibrary/FullAtomModel/Angles2Coords/__init__.py',
						'TorchProteinLibrary/FullAtomModel/Angles2Coords/Angles2Coords.py',
						'TorchProteinLibrary/FullAtomModel/Coords2TypedCoords/__init__.py',
						'TorchProteinLibrary/FullAtomModel/Coords2TypedCoords/Coords2TypedCoords.py',
						'TorchProteinLibrary/FullAtomModel/CoordsTransform/__init__.py',
						'TorchProteinLibrary/FullAtomModel/CoordsTransform/CoordsTransform.py',
						'TorchProteinLibrary/FullAtomModel/PDB2Coords/__init__.py',
						'TorchProteinLibrary/FullAtomModel/PDB2Coords/PDB2Coords.py',
						#ReducedModel
						'TorchProteinLibrary/ReducedModel/Angles2Backbone/__init__.py',
						'TorchProteinLibrary/ReducedModel/Angles2Backbone/Angles2Backbone.py',
						#Volume
						'TorchProteinLibrary/Volume/TypedCoords2Volume/__init__.py',
						'TorchProteinLibrary/Volume/TypedCoords2Volume/TypedCoords2Volume.py',
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
					include_dirs = ['Layers/FullAtomModel',
					'Math'])
	Volume = CUDAExtension('_Volume',
					sources = [
					'Layers/Volume/TypedCoords2Volume/typedcoords2volume_interface.cpp',
					'Layers/Volume/Volume2Xplor/volume2xplor_interface.cpp',
					'Layers/Volume/Kernels.cu',
					'Layers/Volume/main.cpp'],
					include_dirs = ['Layers/Volume'])

	ReducedModel = CUDAExtension('_ReducedModel',
					sources = [
					'Layers/ReducedModel/Angles2Backbone/angles2backbone_interface.cpp',
					'Layers/ReducedModel/cBackboneProteinCUDAKernels.cu',
					'Layers/ReducedModel/main.cpp'],
					include_dirs = ['Layers/ReducedModel'])
	
	setup(	name='TorchProteinLibrary',
			version="0.1",
			ext_modules=[	FullAtomModel, 
							Volume, 
							ReducedModel],
			cmdclass={'build_ext': BuildExtension},
			scripts = PytorchInterace,
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
