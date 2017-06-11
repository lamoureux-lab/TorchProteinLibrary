CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
project(MainProj)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#SET(WITH_OPENMP OFF)
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 7.5 REQUIRED)

IF(DEFINED Torch_INSTALL_INCLUDE)
  SET(TH_INSTALL_INCLUDE ${Torch_INSTALL_INCLUDE})
  SET(TH_INSTALL_LIB ${Torch_INSTALL_LIB})
ENDIF()

INCLUDE_DIRECTORIES(
  ${TH_INSTALL_INCLUDE}/TH
  ${TH_INSTALL_INCLUDE}
)

INCLUDE_DIRECTORIES("/home/lupoglaz/NVIDIA_CUDA-7.5_Samples/common/inc")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES(${TH_INSTALL_LIB})

add_definitions(-std=c++11 -fPIC)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20 -Xcompiler -fPIC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")


#TH_MATH: tensor scaling, rotating and saving to ccp4
include_directories("Math")
add_subdirectory(Math)

#PROTEINLOADER: protein loading layer
include_directories("Layers/PDB2Coords")
add_subdirectory(Layers/PDB2Coords)

#COORDS2VOLUME: projection of coordinates on a grid on gpu
include_directories("Layers/Coords2Volume")
add_subdirectory(Layers/Coords2Volume)

#GLFRAMEWORK: freeglut visualization library
include_directories("GL")
add_subdirectory(GL)

#TENSORVIS: tensor visualization library
include_directories("VolumeVisualization")
add_subdirectory(VolumeVisualization)



#Test tensor visualization using protein loading and visualizing it's density
cuda_add_executable(cpuProteinLoaderTest UnitTests/cpuProteinLoaderTest.cpp)
target_link_libraries (cpuProteinLoaderTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK)

#Test PDB2Coords layer
cuda_add_executable(pdb2CoordsTest UnitTests/pdb2CoordsTest.cpp)
target_link_libraries (pdb2CoordsTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK)

#Test PDB2Coords layer
cuda_add_executable(coords2VolumeTest UnitTests/coords2VolumeTest.cpp)
target_link_libraries (coords2VolumeTest TH PROTEINLOADER MARCHINGCUBES GLFRAMEWORK COORDS2VOLUME)

