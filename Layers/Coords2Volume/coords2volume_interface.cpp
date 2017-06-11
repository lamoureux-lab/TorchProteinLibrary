#include <THC/THC.h>
#include "cCoords2Volume.h"
#include <iostream>
extern THCState *state;
extern "C" {
    int Coords2Volume_forward(  THCudaTensor *gpu_plain_coords, 
                                THCudaIntTensor *gpu_offsets, 
                                THCudaIntTensor *gpu_num_coords_of_type,
                                THCudaTensor *gpu_volume,
                                int num_types,
                                int box_size,
                                float resolution ){
        cCoords2Volume c2v(num_types, box_size, resolution);
        c2v.computeForward(state, gpu_plain_coords, gpu_offsets, gpu_num_coords_of_type, gpu_volume);
    }
}