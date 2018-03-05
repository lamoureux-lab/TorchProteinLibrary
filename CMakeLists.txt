
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
project(MainProj)
option(FAM_ONLY "FAM_ONLY" OFF)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++11 -fPIC -lpthread)

#TORCH package
FIND_PACKAGE(Torch REQUIRED)

IF(DEFINED Torch_INSTALL_INCLUDE)
  SET(TH_INSTALL_INCLUDE ${Torch_INSTALL_INCLUDE})
  SET(TH_INSTALL_LIB ${Torch_INSTALL_LIB})
ENDIF()

INCLUDE_DIRECTORIES(
  ${TH_INSTALL_INCLUDE}/TH
  ${TH_INSTALL_INCLUDE}
)

if(NOT FAM_ONLY)
  # SET(CMAKE_C_COMPILER /usr/bin/gcc-6)
  # SET(CMAKE_CXX_COMPILER /usr/bin/g++-6)
  FIND_PACKAGE(CUDA 9.1 REQUIRED)
  INCLUDE_DIRECTORIES("${CUDA_INCLUDE_DIRS}")

  INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
  LINK_DIRECTORIES(${TH_INSTALL_LIB})

  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

endif(NOT FAM_ONLY)

LINK_DIRECTORIES(${TH_INSTALL_LIB})

#TH_MATH: tensor scaling, rotating and saving to ccp4
include_directories("Math")
add_subdirectory(Math)
if(NOT FAM_ONLY)
  #PROTEINLOADER: protein loading layer
  #include_directories("Layers/PDB2Coords")
  #add_subdirectory(Layers/PDB2Coords)

  #COORDS2VOLUME: projection of coordinates on a grid on gpu
  #include_directories("Layers/Coords2Volume")
  #add_subdirectory(Layers/Coords2Volume)

  #TENSORUPSAMPLE: upsampling a tensor on cpu
  #include_directories("Layers/Upsampling")
  #add_subdirectory(Layers/Upsampling)
endif(NOT FAM_ONLY)

#GLFRAMEWORK: freeglut visualization library
include_directories("GL")
add_subdirectory(GL)

#TENSORVIS: tensor visualization library
include_directories("VolumeVisualization")
add_subdirectory(VolumeVisualization)

if(NOT FAM_ONLY)
 

  # UTILS: some useful utilities
  include_directories("Utils")
  add_subdirectory(Utils)

  #ANGLES2BACKBONE: convert backbone dihedral angles to coordinates
  # include_directories("Layers/Angles2Backbone")
  # add_subdirectory(Layers/Angles2Backbone)

  #CALPHA_INTERNAL_COORDINATES and CALPHARMSD: cuda libraries for C-alpha protein model
  include_directories("Layers/C_alpha_protein")
  add_subdirectory(Layers/C_alpha_protein)
endif(NOT FAM_ONLY)

#FULL-ATOM MODEL
include_directories("Layers/FullAtomModel")
add_subdirectory(Layers/FullAtomModel)

#Remote visualization
include_directories("Visualization")
add_subdirectory(Visualization)



if(NOT FAM_ONLY)
  #Test tensor visualization using protein loading and visualizing it's density
  #cuda_add_executable(cpuProteinLoaderTest UnitTests/cpuProteinLoaderTest.cpp)
  #target_link_libraries (cpuProteinLoaderTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK)

  #Test PDB2Coords layer
  #cuda_add_executable(pdb2CoordsTest UnitTests/pdb2CoordsTest.cpp)
  #target_link_libraries (pdb2CoordsTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK)

  #Test PDB2Coords layer
  #cuda_add_executable(coords2VolumeTest UnitTests/coords2VolumeTest.cpp)
  #target_link_libraries (coords2VolumeTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK COORDS2VOLUME)

  #Test Utils saving 3d tensor to xplor
  #add_executable(densityMapSaveTest UnitTests/densityMapSaveTest.cpp)
  #target_link_libraries (densityMapSaveTest TH UTILS TH_MATH)

  #Test upsampling of tensor on cpu
  #add_executable(tensorUpsampleTest UnitTests/tensorUpsampleTest.cpp)
  #target_link_libraries (tensorUpsampleTest TH TENSORUPSAMPLE TH_MATH)

  #Test backbone computation
  #add_executable(angles2BackboneTest UnitTests/angles2BackboneTest.cpp)
  #target_link_libraries (angles2BackboneTest TH GLFRAMEWORK)


  #Test backbone computation
  add_executable(forces2DanglesTest UnitTests/forces2DanglesTest.cpp Layers/C_alpha_protein/Angles2CoordsAB/cAngles2CoordsAB.cpp Layers/C_alpha_protein/Forces2Dangles/cForces2Dangles.cpp)
  target_link_libraries (forces2DanglesTest TH THC AB_MODEL GLFRAMEWORK)

  #Test backbone computation
  add_executable(Ddist2ForcesTest UnitTests/ddist2ForcesTest.cpp Layers/C_alpha_protein/Angles2CoordsAB/cAngles2CoordsAB.cpp Layers/C_alpha_protein/Forces2Dangles/cForces2Dangles.cpp Layers/C_alpha_protein/Ddist2Forces/cDdist2Forces.cpp)
  target_link_libraries (Ddist2ForcesTest TH THC AB_MODEL GLFRAMEWORK)
endif(NOT FAM_ONLY)

#Test FullAtomModel
add_executable(fullAtomModelTest UnitTests/fullAtomModelTest.cpp)
target_link_libraries (fullAtomModelTest TH_MATH FULL_ATOM_MODEL GLFRAMEWORK)

#Test PDBLoader
add_executable(PDBLoaderTest UnitTests/PDBLoaderTest.cpp)
target_link_libraries (PDBLoaderTest TH_MATH FULL_ATOM_MODEL GLFRAMEWORK)
