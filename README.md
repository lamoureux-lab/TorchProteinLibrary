# TorchProteinLibrary version 0.15
This library pytorch layers for working with protein structures in a differentiable way. We are working on this project and it's bound to change:
there will be interface changes to the current layers, addition of the new ones and code optimizations.

# Requirements
 - GCC >= 7 
 - CUDA >= 10.0
 - PyTorch >= 1.1
 - Python >= 3.5
 - Biopython
 - setuptools

# Installation

Clone the repository:

*git clone https://github.com/lupoglaz/TorchProteinLibrary.git*

then run the following command:

*python setup.py install*

# Contents
The library is structured in the following way:

## FullAtomModel
This module deals with full-atom representation of a protein.
Layers:
- **Angles2Coords**: computes the coordinates of protein atoms, given dihedral angles
- **Coords2TypedCoords**: rearranges coordinates according to predefined atom types 
- **CoordsTransform**: implementations of translation, rotation, centering in a box, random rotation matrix, random translation
- **PDB2Coords**: loading of PDB atomic coordinates

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

**_Volume.Volume2Xplor**: saves volume to XPLOR format


# General design decisions
The library is structured in the following way:
- Layers directory contains c++/cuda implementations
- Each layer has **<layer_name>_ interface.h** and **.cpp** files, that have implementations of functions that are exposed to python
- Each python extension has **main.cpp** file, that contains macros with definitions of exposed functions

We found that these principles provide readability and overall cleaner design.
