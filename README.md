# TorchProteinLibrary
This library contains C++ and CUDA procedures for working with protein structures in a differentiable way. 
They are accompanied by a PyTorch interface.

# Requirements
 - GCC > 4.9
 - CUDA >= 9.0
 - PyTorch >= 0.4.1
 - Python >= 3.5
 - Biopython
 - setuptools

# Installation

Clone the repository:

*git clone https://github.com/lupoglaz/TorchProteinLibrary.git*

then run the following command:

*python setup.py install*

# Contents
## FullAtomModel
This module deals with full-atom representation of a protein.
Layers:
- **Angles2Coords**: computes the coordinates of protein atoms, given dihedral angles
- **Coords2TypedCoords**: rearranges coordinates according to predefined atom types 
- **CoordsTransform**: implementations of translation, rotation, centering in a box, random rotation matrix, random translation
- **PDB2CoordsBiopython**: load PDB atomic coordinates in the same order as in the PDB file
- **PDB2Coords**: load PDB atomic coordinates in the same order as in the output of **Angles2Coords** layer

## ReducedModel
The coarse-grained representation of a protein.
- **Angles2Backbone**: computes the coordinates of protein backbone atoms, given dihedral angles

## RMSD
For now, only contains implementation of differentiable least-RMSD.
Layers:
- **Coords2RMSD**: computes minimum RMSD by optimizing *wrt* translation and rotation of input coordinates

## Volume
Deals with volumetric representation of a protein.
- **TypedCoords2Volume**: computes 3d density maps of coordinates with assigned types
- **Select**: selects cells from a set of volumes at scaled input coordinates
- **VolumeConvolution**: computes correlation of two volumes of equal size

Additional useful function in C++ extension **_Volume**:

**_Volume._Volume2Xplor**: saves volume to XPLOR format


# General design decisions
The library is structured in the following way:
- Layers directory contains c++/cuda implementations
- Each layer has **<layer_name>_ interface.h** and **.cpp** files, that have implementations of functions that are exposed to python
- Each python extension has **main.cpp** file, that contains macros with definitions of exposed functions

We found that these principles provide readability and overall cleaner design.
