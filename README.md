# ProteinClassesLibrary
This library contains C++ and CUDA procedures for working with protein structures in a differentiable way. They are accompanied by the PyTorch interface.

# General design decisions
The library is structures in the following way:
- Cmake file compiles C++ UnitTests and CUDA static libraries
- Each module is a C++ class, that wraps CUDA calls
- Each python module has its own build script, it compiles interface C++ files and links the static libraries compiled at the previos step

We found that these principles provide readability and overall cleaner design.

# Contents
## C-alpha model
The C++ directory is *Layers/C_alpha_protein*.  
The PyTorch modules are *PythonInterface/C_alpha_protein*.  
The list of modules:
- **Angles to coordinates**: computes the coordinates of C-alpha model given the tensor of phi-psi angles
- **Angles to basis**: computes the basis vectors cantered on each C-alpha atom given the tensor of phi-psi angles
- **Coordinates to pairs**: computes the pairwise difference between coordinates given the coordinates
- **Pairs to distributions**: computes the pairwise distance distributions of C-alpha atoms given the pairwise coordinates differences

## PDB
These layers are not designed to work together with C-alpha model layers. The format of the coordinates is different from the C-alpha atoms coordinates. The gradients are not implemented for these modules.
The list of modules:
- **PDB to coordinates**: this layer reads PDB file, assigns atoms 11 types and rearranges the data
- **Coordinates to volume**: applies random uniform rotation and translation and projects the coordinates data on the corresponding 3D grids.

## Visualization
This module provide a way to visualize all the 11 atomic densities simultaneously using the marching cubes algorithm.

## Utils
This module helps saving the density maps to the XPlor format that is readable by PyMOL.

# Installation
1. Create *build* directory
2. Change to the *build* directory, run *cmake* and *make*
3. Go to PythonInterface and compile the modules you need using the *build.py* scripts
4. The modules can then be imported from the root directory of the library
