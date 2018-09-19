
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
  ${TH_INSTALL_INCLUDE}/THC
  ${TH_INSTALL_INCLUDE}
)

LINK_DIRECTORIES(${TH_INSTALL_LIB})


#CUDA
if(GRAHAM)
  FIND_PACKAGE(CUDA 8.0 REQUIRED)
endif(GRAHAM)
if(NOT GRAHAM)
  FIND_PACKAGE(CUDA 9.1 REQUIRED)
endif(NOT GRAHAM)

INCLUDE_DIRECTORIES("${CUDA_INCLUDE_DIRS}")

LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC -arch=sm_52  -lineinfo ")

#CPU MATH
include_directories("Math")
add_subdirectory(Math)

#BACKBONE_INTERNAL_COORDINATES and CALPHARMSD: cuda libraries for backbone protein model
include_directories("Layers/ReducedModel")
add_subdirectory(Layers/ReducedModel)

include_directories("Layers/Volume")
add_subdirectory(Layers/Volume)

#FULL-ATOM MODEL
include_directories("Layers/FullAtomModel")
add_subdirectory(Layers/FullAtomModel)



add_executable(Angles2BackboneProfile UnitTests/Angles2Backbone/angles2BackboneProfile.cpp)
target_link_libraries (Angles2BackboneProfile TH THC BACKBONE_INTERNAL_COORDINATES)

add_executable(Angles2BackboneMemtest UnitTests/Angles2Backbone/angles2BackboneMemtest.cpp)
target_link_libraries (Angles2BackboneMemtest TH THC BACKBONE_INTERNAL_COORDINATES)

cuda_add_executable(volumeConvolutionProfile UnitTests/volumeConvolutionProfile.cpp)
target_link_libraries (volumeConvolutionProfile TH THC VOLUMECONV)

find_library(CUFFT_LIBRARY cufft HINTS /usr/local/cuda-9.1/lib64)
cuda_add_executable(cudaConvolutions UnitTests/cudaConvolutions.cpp)
target_link_libraries (cudaConvolutions TH THC VOLUMECONV ${CUFFT_LIBRARY})
