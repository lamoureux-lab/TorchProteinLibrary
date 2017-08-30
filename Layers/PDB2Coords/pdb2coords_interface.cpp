#include <THC/THC.h>
#include "cProteinLoader.h"
#include <iostream>
#include <string>
extern THCState *state;
extern "C" {
    int PDB2Coords_forward( const char* filename, 
                            THCudaTensor *gpu_plain_coords, 
                            THCudaIntTensor *gpu_offsets, 
                            THCudaIntTensor *gpu_num_coords_of_type, 
                            int spatial_dim, 
                            int resolution, 
                            int rotate, 
                            int shift ){
        cProteinLoader pL;
        pL.pdb2Coords(state, filename, gpu_plain_coords, gpu_offsets, gpu_num_coords_of_type, spatial_dim, resolution, rotate, shift);
    }

}